# engines/linops/base.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Tuple, Optional
import jax
import jax.numpy as jnp
from jax import Array

@jax.tree_util.register_pytree_node_class
@dataclass(init=False)
class LinearOp:
    shape: Tuple[int, int]
    # Core
    _mv: Callable[[Array], Array]                # v -> A v
    _rmv: Optional[Callable[[Array], Array]] = None  # v -> A^T v (optional)
    _to_dense: Optional[Callable[[], Array]] = None

    def __init__(self, shape, mv, rmv=None, to_dense=None):
        self.shape = shape
        self._mv = mv
        self._rmv = rmv
        self._to_dense = to_dense

    # Public API
    def mv(self, v: Array) -> Array: return self._mv(v)
    
    def rmv(self, v: Array) -> Array:
        if self._rmv is None:
            # symmetric fallback if square
            assert self.shape[0] == self.shape[1], "rmv requires square or explicit rmv"
            return self._mv(v)
        return self._rmv(v)
    def matmul(self, X: Array) -> Array:  # A @ X
        return jax.vmap(self._mv, in_axes=1, out_axes=1)(X)

    def to_dense(self) -> Array:
        if self._to_dense is not None:
            return self._to_dense()
        # generic: apply to basis (ok for small n or debugging)
        n, m = self.shape
        I = jnp.eye(m)
        return jax.vmap(self._mv, in_axes=1, out_axes=1)(I)

    # PyTree
    def tree_flatten(self):
        # functions are static (aux); nothing differentiable by default
        return (), (self.shape, self._mv, self._rmv, self._to_dense)
    @classmethod
    def tree_unflatten(cls, aux, ch):
        shape, mv, rmv, to_dense = aux
        return cls(shape, mv, rmv, to_dense)


# Convenience constructors
def DenseOp(A: Array) -> LinearOp:
    n, m = A.shape
    return LinearOp((n, m), mv=lambda v: A @ v, rmv=lambda v: A.T @ v, to_dense=lambda: A)

def IdentityOp(n: int, dtype) -> LinearOp:
    return LinearOp((n, n), mv=lambda v: v, rmv=lambda v: v, to_dense=lambda: jnp.eye(n, dtype))

def ScaledIdentityOp(n: int, scale: Array, dtype) -> LinearOp:
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
