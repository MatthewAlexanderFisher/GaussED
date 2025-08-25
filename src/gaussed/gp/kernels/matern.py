from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Callable, Protocol
import jax, jax.numpy as jnp
from jax import Array

from gaussed.domains.base import Domain
from gaussed.utils.constraints import Positive, Transform

# --- Matérn Kernels with ν = 3/2, 5/2, 7/2 ----------------------------------------
@jax.tree_util.register_pytree_node_class
@dataclass
class MaternParams:
    lengthscale: Array
    amplitude: Array
    nu: float

    def tree_flatten(self):
        return (self.lengthscale, self.amplitude), (self.nu,)
    
    @classmethod
    def tree_unflatten(cls, aux, ch):
        (nu,) = aux
        raw_ell, raw_sigma2 = ch
        return cls(raw_ell, raw_sigma2, nu)
    
    @classmethod
    def axes(cls, ell_axis, s2_axis, *, nu: float):
        obj = object.__new__(cls)
        obj.lengthscale = ell_axis; obj.amplitude = s2_axis; obj.nu = nu
        return obj

@jax.tree_util.register_pytree_node_class
@dataclass
class MaternKernel:
    params: MaternParams  # supports ν = 3/2, 5/2, 7/2 via params.nu
    ell_transform: Transform = field(default_factory=Positive)
    sigma2_transform: Transform = field(default_factory=Positive)

    def __call__(self, x, y, domain):

        ell = self.ell_transform.forward(self.params.lengthscale)
        s2  = self.sigma2_transform.forward(self.params.amplitude)

        r = domain.pairwise_geometry(x, y) / ell
        nu = self.params.nu
        if nu == 1.5:
            a = jnp.sqrt(3.0); poly = 1.0 + a*r
            return s2 * poly * jnp.exp(-a*r)
        elif nu == 2.5:
            a = jnp.sqrt(5.0); poly = 1.0 + a*r + (5.0/3.0)*r*r
            return s2 * poly * jnp.exp(-a*r)
        elif nu == 3.5:  # 7/2
            a = jnp.sqrt(7.0); r2 = r*r; r3 = r2*r
            poly = 1.0 + a*r + (14.0/5.0)*r2 + (7.0*a/15.0)*r3
            return s2 * poly * jnp.exp(-a*r)
        else:
            raise ValueError("Supported ν: 3/2, 5/2, 7/2 only.")
    
    # Make transforms static (aux), params a leaf
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
    def axes(cls, params_axis: MaternParams):
        obj = object.__new__(cls)
        obj.params = params_axis
        # Transforms are static; reuse defaults
        obj.ell_transform = Positive()
        obj.sigma2_transform = Positive()
        return obj
