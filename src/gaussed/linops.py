# engines/linops/base.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Tuple, Optional
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import DTypeLike

from gaussed.domains.base import Domain
from gaussed.types import LinearLike

@jax.tree_util.register_pytree_node_class
@dataclass(init=False)
class LinearOp:
    """Lightweight linear operator with optional rmatvec and optional dense materialisation.

    Shape convention: (n_rows, n_cols) = shape.
    mv:  (m,) or (m,k) -> (n,) or (n,k)
    rmv: (n,) or (n,k) -> (m,) or (m,k)
    """
    # --- differentiable children ---
    _A: Optional[Array]                 # dense matrix, if available

    # --- static aux (non-diff) ---
    _shape: Tuple[int, ...]             # only used when _A is None
    _mv: Optional[Callable[[Array], Array]]          # v -> A v  (vector OR thin matrix)
    _rmv: Optional[Callable[[Array], Array]]         # v -> A^T v
    _to_dense: Optional[Callable[[], Array]]         # () -> dense matrix
    dtype: Optional[DTypeLike] = None  # if provided, used for to_dense   

    def __init__(
        self,
        shape: Optional[Tuple[int, ...]] = None,
        *,
        A: Optional[Array] = None,
        mv: Optional[Callable[[Array], Array]] = None,
        rmv: Optional[Callable[[Array], Array]] = None,
        to_dense: Optional[Callable[[], Array]] = None,
    ):
        # If dense A is supplied, it defines shape & fast paths.
        if A is not None:
            n, m = A.shape
            self._A = A
            self._shape = (n, m)
            self._mv = mv
            self._rmv = rmv
            self._to_dense = to_dense
            self.dtype = A.dtype
            return

        if shape is None:
            raise ValueError("LinearOp: provide either dense A or a shape with mv/rmv/to_dense.")
        self._A = None
        self._shape = shape
        self._mv = mv
        self._rmv = rmv
        self._to_dense = to_dense

        if self._mv is None and self._to_dense is None:
            raise ValueError("LinearOp: with no dense A, you must provide mv or to_dense.")

    # -----------------------------
    # Public API
    # -----------------------------
    @property
    def shape(self) -> Tuple[int, ...]:
        return self._A.shape if self._A is not None else self._shape

    def _mv_dense(self, X: Array) -> Array:
        # Supports (m,) and (m,k)
        return self._A @ X  # type: ignore[operator]

    def _rmv_dense(self, Y: Array) -> Array:
        return self._A.T @ Y  # type: ignore[operator]

    def _mv_fallback(self, X: Array) -> Array:
        if self._mv is not None:
            # Support (m,) and (m,k) for user-supplied mv
            return self._mv(X)
        # No mv: try to_dense -> vmapped basis
        n, m = self.shape
        if X.ndim == 1:
            # apply via to_dense
            return self.to_dense() @ X
        # (m,k): vmapped mv constructed from to_dense
        return self.to_dense() @ X

    def _rmv_fallback(self, Y: Array) -> Array:
        if self._rmv is not None:
            return self._rmv(Y)
        # If no explicit rmv: allow only if square; use mv on transposed (dense) or mv fallback
        n, m = self.shape
        if n != m:
            raise ValueError("rmv requires either an explicit rmv or a square operator.")
        # Use dense if possible; otherwise use to_dense transpose
        return self.to_dense().T @ Y

    def mv(self, X: Array) -> Array:
        """Matvec: returns A @ X (vector or thin matrix)."""
        if self._A is not None:
            return self._mv_dense(X)
        # If user supplied mv, use it; otherwise materialise once via to_dense
        if self._mv is not None:
            return self._mv(X)
        return self._mv_fallback(X)

    def rmv(self, Y: Array) -> Array:
        """R-matvec: returns A^T @ Y (vector or thin matrix)."""
        if self._A is not None:
            return self._rmv_dense(Y)
        if self._rmv is not None:
            return self._rmv(Y)
        return self._rmv_fallback(Y)

    def matmul(self, X: Array) -> Array:
        """Matrix multiplication with a dense X on the right: A @ X."""
        # Just a synonym for mv with explicit vector/matrix handling.
        return self.mv(X)

    def to_dense(self) -> Array:
        """Materialise a dense matrix representation."""
        if self._A is not None:
            return self._A
        if self._to_dense is not None:
            return self._to_dense()
        # Generic (debug/small n): apply to basis of R^m
        n, m = self.shape
        I = jnp.eye(m, dtype=jnp.result_type(0.0))
        # Column-wise mv; if only rmv is provided, use (A^T)^T trick via to_dense above
        if self._mv is not None:
            return jax.vmap(self._mv, in_axes=1, out_axes=1)(I)
        # Last resort: build via rmv on basis of R^n and transpose
        e = jnp.eye(n, dtype=I.dtype)
        return jax.vmap(self.rmv, in_axes=1, out_axes=1)(e).T

    @classmethod
    def from_dense(cls, A: Array) -> LinearOp:
        return cls(A=A)
    
    # -----------------------------
    # Transpose
    # -----------------------------
    @property
    def T(self) -> "LinearOp":
        """Transpose operator with preserved dense fast path if available."""
        n, m = self.shape
        if self._A is not None:
            return LinearOp(A=self._A.T)
        # Swap mv/rmv if both present; otherwise define via to_dense transpose
        mv_T: Optional[Callable[[Array], Array]]
        rmv_T: Optional[Callable[[Array], Array]]
        to_dense_T: Optional[Callable[[], Array]]

        if (self._mv is not None) or (self._rmv is not None):
            mv_T = self._rmv if self._rmv is not None else None
            rmv_T = self._mv if self._mv is not None else None
            def _to_dense_T() -> Array:
                return self.to_dense().T
            to_dense_T = _to_dense_T
            return LinearOp((m, n), mv=mv_T, rmv=rmv_T, to_dense=to_dense_T)

        def _to_dense_T() -> Array:
            return self.to_dense().T
        return LinearOp((m, n), to_dense=_to_dense_T)

    # -----------------------------
    # PyTree plumbing
    # -----------------------------
    def tree_flatten(self):
        # Arrays are children; functions & shape are aux.
        children = (self._A,)
        aux = (self._shape, self._mv, self._rmv, self._to_dense)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (shape, mv, rmv, to_dense) = aux
        (A,) = children
        if A is not None:
            return cls(A=A, mv=mv, rmv=rmv, to_dense=to_dense)
        return cls(shape, mv=mv, rmv=rmv, to_dense=to_dense)


# Convenience constructors
def DenseOp(A: Array) -> LinearOp:
    n, m = A.shape
    return LinearOp((n, m), mv=lambda v: A @ v, rmv=lambda v: A.T @ v, to_dense=lambda: A)

def IdentityOp(n: int, dtype: DTypeLike | None) -> LinearOp:
    return LinearOp((n, n), mv=lambda v: v, rmv=lambda v: v, to_dense=lambda: jnp.eye(n, dtype))

def ScaledIdentityOp(n: int, scale: Array, dtype: DTypeLike | None) -> LinearOp:
    s = jnp.asarray(scale)
    return LinearOp((n, n), mv=lambda v: s * v, rmv=lambda v: s * v,
                    to_dense=lambda: jnp.eye(n, dtype) * s)

def DiagOp(d: Array) -> LinearOp:
    d = jnp.asarray(d)
    n = d.shape[0]
    return LinearOp((n, n), mv=lambda v: d * v, rmv=lambda v: d * v,
                    to_dense=lambda: jnp.diag(d))

def SumOp(A: LinearOp, B: LinearOp) -> LinearOp:
    assert A.shape == B.shape
    if A._A is not None and B._A is not None:
        return LinearOp(A.shape, A = A._A + B._A)
    return LinearOp(A.shape, mv=lambda v: A.mv(v) + B.mv(v),
                    rmv=lambda v: A.rmv(v) + B.rmv(v),
                    to_dense=lambda: A.to_dense() + B.to_dense())

def BlockDiagOp(blocks: tuple[LinearOp, ...]) -> LinearOp:
    sizes = [b.shape[0] for b in blocks]
    n = sum(sizes)
    def mv(v):
        outs = []
        start = 0
        for b, m in zip(blocks, sizes):
            outs.append(b.mv(v[start:start+m]))
            start += m
        return jnp.concatenate(outs, axis=0)
    def to_dense():
        mats = [b.to_dense() for b in blocks]
        return jax.scipy.linalg.block_diag(*mats)
    return LinearOp((n, n), mv=mv, rmv=mv, to_dense=to_dense)

def AsLinearOp(A: LinearLike) -> LinearOp:
    return A if isinstance(A, LinearOp) else DenseOp(A)

def materialise_dense(A: LinearLike) -> Array:
    return A if isinstance(A, Array) else A.to_dense()


#==== Gram Matrix Constructors ====
def _prod(shape: Tuple[int, ...]) -> int:
    p = 1
    for s in shape: p *= int(s)
    return p


def TensorGramOp(k, X: Array, Y: Array, domain: Domain, to_dense: bool = False):
    """
    Build a LinearOp for K(X, Y) where k(x,y) has shape left+right.
    Operator shape: (nx*L, ny*R), L=prod(left), R=prod(right).
    """
    nx = X.shape[0]; ny = Y.shape[0]
    left  = tuple(getattr(k, "left_shape", ()))
    right = tuple(getattr(k, "right_shape", ()))
    L = _prod(left) if left else 1
    R = _prod(right) if right else 1

    def mv(v_flat: Array) -> Array:
        V = v_flat.reshape(ny, R)
        # For fixed x, contract over (ny,right) with V
        def row_apply(x):
            KxY = jax.vmap(lambda y: k(x, y, domain))(Y)    # (ny, *left, *right)
            KxY = KxY.reshape(ny, L, R)
            return jnp.einsum("ylr,yr->yl", KxY, V)         # (L,)
        out = jax.vmap(row_apply)(X)                         # (nx, L)
        return out.reshape(nx * L)

    def rmv(w_flat: Array) -> Array:
        W = w_flat.reshape(nx, L)
        def col_apply(y):
            KYx = jax.vmap(lambda x: k(x, y, domain))(X)    # (nx, *left, *right)
            KYx = KYx.reshape(nx, L, R)
            return jnp.einsum("xlr,xl->xr", KYx, W)         # (R,)
        out = jax.vmap(col_apply)(Y)                         # (ny, R)
        return out.reshape(ny * R)

    def dense():
        Kxy = jax.vmap(lambda x: jax.vmap(lambda y: k(x, y, domain))(Y))(X)  # (nx,ny,*l,*r)
        Kxy = Kxy.reshape(nx, ny, L, R)
        return jnp.transpose(Kxy, (0, 2, 1, 3)).reshape(nx * L, ny * R)

    return LinearOp((nx * L, ny * R), mv=mv, rmv=rmv, to_dense=(dense if to_dense else None))
