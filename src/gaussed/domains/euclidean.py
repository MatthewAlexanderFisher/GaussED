from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Callable, Any
import jax
import jax.numpy as jnp
from jax import tree_util as jtu
from jax import Array

from gaussed.utils.geometry import pairwise_r
from gaussed.utils.shape_helpers import _ensure_n_by_d

@jax.tree_util.register_pytree_node_class
@dataclass(init=False)
class Euclidean:
    shape: Tuple[int, ...]
    eps: float

    def __init__(self, shape: Tuple[int, ...], eps: float = 0.0):
        self.shape = tuple(shape)
        self.eps = float(eps)

    # API
    @property
    def event_shape(self): return self.shape

    def project(self, x: Array) -> Array: return x

    def pairwise_geometry(self, X, Y, *, robust: bool = False):
        if robust:
            X = _ensure_n_by_d(X); Y = _ensure_n_by_d(Y)
            def row_map(x): return jax.vmap(lambda y: pairwise_r(x, y, self.eps))(Y)
            return jax.vmap(row_map)(X)
        # fast path
        X = _ensure_n_by_d(X); Y = _ensure_n_by_d(Y)
        XX = jnp.sum(X*X, axis=1, keepdims=True)
        YY = jnp.sum(Y*Y, axis=1, keepdims=True).T
        sq = jnp.clip(XX + YY - 2.0 * (X @ Y.T), a_min=0.0)
        return jnp.sqrt(sq + self.eps)

    
    def default_nodes(self, n: int) -> Optional[Array]: return None

    # pytree
    def tree_flatten(self):           # all-static config ⇒ put in aux
        return (), (self.shape, self.eps)
    
    @classmethod
    def tree_unflatten(cls, aux, children):
        shape, eps = aux; return cls(shape, eps)

    # vmap axes
    @classmethod
    def axes(cls, shape_axis, eps_axis):
        obj = object.__new__(cls); obj.shape = shape_axis; obj.eps = eps_axis; return obj
