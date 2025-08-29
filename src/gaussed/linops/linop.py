# engines/linops/base.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Tuple, Optional, Protocol, Any
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import DTypeLike

from gaussed.domains.base import Domain
from gaussed.types import LinearLike
from gaussed.utils.shape_helpers import _prod, _as_2d, _restore, pack_kernel

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

    def diag(self) -> Array:
        """
        Return the main diagonal of A as shape (min(n_rows, n_cols),).
        Uses dense fast-path if available; otherwise applies a single
        batched mv/rmv with a thin identity to avoid full materialisation.
        """
        n, m = self.shape
        k = min(n, m)

        # Dense fast path
        if self._A is not None:
            return jnp.diag(self._A)  # already length k

        # Helper to pick a basis dtype (avoid forcing to_dense just for dtype)
        basis_dtype = self.dtype if self.dtype is not None else jnp.result_type(0.0)

        # If both mv and rmv exist: choose the smaller basis dimension
        if (self._mv is not None) and (self._rmv is not None):
            if m <= n:
                # Use mv on first k columns of the identity in R^m
                E = jnp.eye(m, k, dtype=basis_dtype)      # (m, k)
                Y = self.mv(E)                             # (n, k) = A[:, :k]
                return jnp.diagonal(Y)                     # (k,)
            else:
                # Use rmv on first k columns of the identity in R^n
                F = jnp.eye(n, k, dtype=basis_dtype)      # (n, k)
                Z = self.rmv(F)                            # (m, k) = (A^T)[:, :k]
                return jnp.diagonal(Z)                     # (k,)

        # Only mv available
        if self._mv is not None:
            E = jnp.eye(m, k, dtype=basis_dtype)          # (m, k)
            Y = self.mv(E)                                 # (n, k)
            return jnp.diagonal(Y)

        # Only rmv available
        if self._rmv is not None:
            F = jnp.eye(n, k, dtype=basis_dtype)          # (n, k)
            Z = self.rmv(F)                                # (m, k)
            return jnp.diagonal(Z)

        # Fallback: materialise
        return jnp.diag(self.to_dense())

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
    out_shape = A.shape[2:]
    A_flat = pack_kernel(A, out_shape)

    return LinearOp.from_dense(A_flat)

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

def AsLinearOp(A: "LinearLike") -> LinearOp:
    return A if isinstance(A, LinearOp) else DenseOp(A)

def materialise_dense(A: "LinearLike") -> Array:
    return A if isinstance(A, Array) else A.to_dense()

