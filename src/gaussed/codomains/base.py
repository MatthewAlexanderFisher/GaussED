from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Callable, Any
import jax
import jax.numpy as jnp
from jax import tree_util as jtu
from jax import Array

from gaussed.utils.shape_helpers import _prod

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Codomain:
    output_shape: Tuple[int, ...]  # e.g. (1,) scalar, (p,), (p,q), ...

    def __init__(self, output_shape: Optional[Tuple[int, ...]]):
        object.__setattr__(self, "output_shape", tuple(output_shape) if output_shape is not None else (1,))

    # Alias, if you still use the old name elsewhere
    @property
    def event_shape(self) -> Tuple[int, ...]:
        return self.output_shape

    # ---- Mean enforcement: (n,*E) ----
    def ensure_mean_outputs(self, Y: Array, n: int) -> Array:
        Y = jnp.asarray(Y)
        E = self.output_shape
        if Y.ndim == len(E):                 # (*E,) -> (1,*E)
            return Y.reshape((1, *E))
        if E == (1,) and Y.ndim == 1:        # (n,) -> (n,1)
            return Y[:, None]
        if Y.ndim == 1 + len(E) and Y.shape[1:] == E:
            return Y
        # permissive reshape if sizes match
        if Y.size == n * _prod(E):
            return Y.reshape((n, *E))
        raise ValueError(f"Mean returned {Y.shape}, expected (n,{E})")

    # ---- Kernel block enforcement: (n_x, n_y, *L, *R) ----
    def enforce_kernel_shape(
        self,
        K: Array,
        n_x: int,
        n_y: int,
        left_shape: Tuple[int, ...],
        right_shape: Tuple[int, ...],
    ) -> Array:
        K = jnp.asarray(K)
        want_nd = 2 + len(left_shape) + len(right_shape)
        if K.ndim == want_nd:
            return K
        if K.ndim == 2 and not left_shape and not right_shape:
            return K
        if K.ndim == 0 and n_x == 1 and n_y == 1 and not left_shape and not right_shape:
            return K.reshape(1, 1)
        if K.shape == (*left_shape, *right_shape) and n_x == 1 and n_y == 1:
            return K.reshape((1, 1, *left_shape, *right_shape))
        size_ok = (K.size == n_x * n_y * _prod(left_shape) * _prod(right_shape))
        if size_ok:
            return K.reshape((n_x, n_y, *left_shape, *right_shape))
        raise ValueError(
            f"Kernel returned {K.shape}; expected (n_x,n_y,{left_shape},{right_shape})"
        )

    # ---- Flatten/unflatten event dims ----
    def flatten_events(self, Y: Array) -> Array:
        Y = jnp.asarray(Y)
        if Y.ndim == 1: return Y
        n = Y.shape[0]; e = _prod(Y.shape[1:])
        return Y.reshape(n * e)

    def unflatten_events(self, y: Array, n: int) -> Array:
        return jnp.asarray(y).reshape(n, *self.output_shape)

    # pytree
    def tree_flatten(self): return (), (self.output_shape,)
    @classmethod
    def tree_unflatten(cls, aux, ch): (out_shape,) = aux; return cls(out_shape)
    @classmethod
    def axes(cls, axis: Tuple[int, ...]) -> "Codomain":
        obj = object.__new__(cls); object.__setattr__(obj, "output_shape", tuple(axis)); return obj
