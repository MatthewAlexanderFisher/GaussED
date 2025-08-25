from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Callable, Any
import jax
import jax.numpy as jnp
from jax import tree_util as jtu
from jax import Array

from gaussed.domains.base import Domain
from gaussed.utils.geometry import pairwise_r

@jax.tree_util.register_pytree_node_class
@dataclass(init=False)
class Rectangle:
    bounds: Array  # (d,2)
    eps: float

    def __init__(self, bounds: Array, eps: float = 0.0):
        self.bounds = jnp.asarray(bounds)
        self.eps = float(eps)

    @property
    def event_shape(self): return (self.bounds.shape[0],)

    def project(self, x: Array) -> Array:
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return jnp.clip(x, lo, hi)

    def pairwise_geometry(self, x: Array, y: Array) -> Array:
        return pairwise_r(self.project(x), self.project(y), self.eps)

    def default_nodes(self, n: int) -> Optional[Array]:
        d = self.bounds.shape[0]
        k = jnp.ceil(n ** (1/d)).astype(int)
        grids = [jnp.linspace(self.bounds[i,0], self.bounds[i,1], int(k)) for i in range(d)]
        mesh = jnp.stack(jnp.meshgrid(*grids, indexing="ij"), axis=-1).reshape(-1, d)
        return mesh[:n]

    def tree_flatten(self):
        return (self.bounds,), (self.eps,)
    
    @classmethod
    def tree_unflatten(cls, aux, children):
        (eps,) = aux; (bounds,) = children; return cls(bounds, eps)

    @classmethod
    def axes(cls, bounds_axis, eps_axis):
        obj = object.__new__(cls); obj.bounds = bounds_axis; obj.eps = eps_axis; return obj
