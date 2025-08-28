# engines/linops/base.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Tuple, Optional
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import DTypeLike

from gaussed.domains.base import Domain

@jax.tree_util.register_pytree_node_class
@dataclass(init=False)
class LinearOp:
    """Lightweight linear operator with optional rmatvec and optional dense materialisation.
    
    Shape convention: (_shape[0], _shape[1]) = (n_rows, n_cols).
    mv:  (m,) or (m,k)  -> (n,) or (n,k)
    rmv: (n,) or (n,k)  -> (m,) or (m,k)
    """
    _shape: Tuple[int, int]
    _mv: Callable[[Array], Array]                 # v -> A v
    _rmv: Optional[Callable[[Array], Array]] = None  # v -> A^T v (optional)
    _to_dense: Optional[Callable[[], Array]] = None  # () -> dense matrix

    def __init__(
        self,
        shape: Tuple[int, int],
        mv: Callable[[Array], Array],
        rmv: Optional[Callable[[Array], Array]] = None,
        to_dense: Optional[Callable[[], Array]] = None,
    ):
        self._shape = shape
        self._mv = mv
        self._rmv = rmv
        self._to_dense = to_dense

    # -----------------------------
    # Public API
    # -----------------------------
    @property
    def shape(self) -> Tuple[int, int]:
        return self._shape

    def mv(self, v: Array) -> Array:
        """Matvec: returns A @ v (vectors or thin matrices)."""
        return self._mv(v)

    def rmv(self, v: Array) -> Array:
        """R-matvec: returns A^T @ v. Falls back to mv only if square and rmv is not provided."""
        if self._rmv is not None:
            return self._rmv(v)
        # Fallback allowed only for square operators
        n, m = self._shape
        if n != m:
            raise ValueError("rmv requires either an explicit rmv or a square operator to use mv fallback.")
        return self._mv(v)

    def matmul(self, X: Array) -> Array:
        """Matrix multiplication with a dense X on the right: A @ X."""
        # Support both vector (m,) and matrix (m,k)
        if X.ndim == 1:
            return self._mv(X)
        return jax.vmap(self._mv, in_axes=1, out_axes=1)(X)

    def to_dense(self) -> Array:
        """Materialise a dense matrix representation."""
        if self._to_dense is not None:
            return self._to_dense()
        # Generic (debug/small n): apply to basis of R^m
        n, m = self._shape
        I = jnp.eye(m, dtype=jnp.float32)
        return jax.vmap(self._mv, in_axes=1, out_axes=1)(I)

    # -----------------------------
    # Transpose
    # -----------------------------
    @property
    def T(self) -> "LinearOp":
        """Transpose operator: swaps mv <-> rmv; if rmv is missing, allowed only if square."""
        n, m = self._shape

        if self._rmv is None:
            if n != m:
                raise ValueError("Cannot form transpose without rmv for a non-square operator.")
            mv_T = self._mv
            rmv_T = self._mv
        else:
            mv_T = self._rmv
            rmv_T = self._mv

        # Make the type of to_dense_T explicit so None is allowed.
        to_dense_T: Optional[Callable[[], Array]] = None
        if self._to_dense is not None:
            def _to_dense_T() -> Array:
                return self.to_dense().T
            to_dense_T = _to_dense_T

        return LinearOp((m, n), mv=mv_T, rmv=rmv_T, to_dense=to_dense_T)

    # -----------------------------
    # PyTree plumbing
    # -----------------------------
    def tree_flatten(self):
        # Functions are static (aux), arrays/params would go in children.
        children = ()
        aux = (self._shape, self._mv, self._rmv, self._to_dense)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        shape, mv, rmv, to_dense = aux
        return cls(shape, mv, rmv, to_dense)

    # -----------------------------
    # Convenience constructors
    # -----------------------------
    @classmethod
    def from_dense(cls, A: Array) -> LinearOp:
        """Build a LinearOp from a dense array."""
        n, m = A.shape
        def mv(v: Array) -> Array:
            return A @ v
        def rmv(v: Array) -> Array:
            return A.T @ v
        def to_dense() -> Array:
            return A
        return cls((n, m), mv=mv, rmv=rmv, to_dense=to_dense)


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

def AsLinearOp(A: Array | LinearOp) -> LinearOp:
    return A if isinstance(A, LinearOp) else DenseOp(A)


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
