from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Callable, Any
import jax
from jax import Array

from gaussed.domains.base import Domain
from gaussed.utils.geometry import pairwise_r

@jax.tree_util.register_pytree_node_class
@dataclass(init=False)
class EmbeddedManifold:
    phi: Callable[[Array], Array]     # static
    intrinsic_shape: Tuple[int, ...]
    eps: float
    use_geodesic: bool

    def __init__(self, phi, intrinsic_shape: Tuple[int, ...], eps: float = 0.0, use_geodesic: bool = False):
        self.phi = phi
        self.intrinsic_shape = tuple(intrinsic_shape)
        self.eps = float(eps)
        self.use_geodesic = bool(use_geodesic)

    @property
    def event_shape(self): return self.intrinsic_shape
    def project(self, u: Array) -> Array: return u
    def pairwise_geometry(self, u: Array, v: Array) -> Array:
        x, y = self.phi(self.project(u)), self.phi(self.project(v))
        return pairwise_r(x, y, self.eps)

    def default_nodes(self, n: int) -> Optional[Array]: return None

    def tree_flatten(self):
        # no dynamic leaves; keep everything static in aux (incl. callable)
        return (), (self.phi, self.intrinsic_shape, self.eps, self.use_geodesic)
    @classmethod
    def tree_unflatten(cls, aux, children):
        phi, ishape, eps, ug = aux; return cls(phi, ishape, eps, ug)

    @classmethod
    def axes(cls, phi_axis, ishape_axis, eps_axis, ug_axis):
        obj = object.__new__(cls)
        obj.phi = phi_axis; obj.intrinsic_shape = ishape_axis; obj.eps = eps_axis; obj.use_geodesic = ug_axis
        return obj
