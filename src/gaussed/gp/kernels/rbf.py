from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Callable
import jax, jax.numpy as jnp
from jax import Array

from gaussed.domains.base import Domain
from gaussed.utils.constraints import Positive, Transform


# --- RBF / Squared-Exponential / Gaussian ------------------------------------
@jax.tree_util.register_pytree_node_class
@dataclass
class RBFParams:
    raw_ell: Array          # unconstrained
    raw_sigma2: Array

    def tree_flatten(self): 
        return (self.raw_ell, self.raw_sigma2), None
    
    @classmethod
    def tree_unflatten(cls, aux, ch): 
        return cls(*ch)

    @classmethod
    def axes(cls, ell_axis, s2_axis):
        obj = object.__new__(cls)
        obj.raw_ell = ell_axis
        obj.raw_sigma2 = s2_axis
        return obj

@jax.tree_util.register_pytree_node_class
@dataclass
class RBF:
    params: RBFParams
    ell_transform: Transform = field(default_factory=Positive)
    sigma2_transform: Transform = field(default_factory=Positive)

    def __call__(self, x: Array, y: Array, domain: Domain) -> Array:
        ell = self.ell_transform.forward(self.params.raw_ell)
        s2 = self.sigma2_transform.forward(self.params.raw_sigma2)

        r = domain.pairwise_geometry(x, y) / (ell + 1e-12)
        return s2 * jnp.exp(-0.5 * r * r)

    def tree_flatten(self):
        children = (self.params,)
        aux = (self.ell_transform, self.sigma2_transform)
        return children, aux
    
    @classmethod
    def tree_unflatten(cls, aux, children):
        ell_t, s2_t = aux
        (params,) = children
        obj = cls(params, ell_t, s2_t)
        return obj

    @classmethod
    def axes(cls, params_axis: RBFParams):
        obj = object.__new__(cls)
        obj.params = params_axis
        # Transforms are static; reuse defaults
        obj.ell_transform = Positive()
        obj.sigma2_transform = Positive()
        return obj