from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Callable, Protocol
import jax, jax.numpy as jnp
from jax import Array

from gaussed.domains.base import Domain
from gaussed.utils.constraints import Positive, Transform
from gaussed.gp.kernels.base import Kernel

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
    lengthscale_transform: Transform = field(default_factory=Positive)
    amplitude_transform: Transform = field(default_factory=Positive)

    # tensor-kernel interface (static; not part of pytree children)
    left_shape: Tuple[int, ...] = field(default_factory=tuple)
    right_shape: Tuple[int, ...] = field(default_factory=tuple)

    def pair(self, x, y, domain):

        lengthscale, amplitude = self.get_transformed_params()

        r = domain.pairwise_geometry(x, y) / lengthscale
        nu = self.params.nu
        if nu == 1.5:
            a = jnp.sqrt(3.0); poly = 1.0 + a*r
            return amplitude * poly * jnp.exp(-a*r)
        elif nu == 2.5:
            a = jnp.sqrt(5.0); poly = 1.0 + a*r + (5.0/3.0)*r*r
            return amplitude * poly * jnp.exp(-a*r)
        elif nu == 3.5:  # 7/2
            a = jnp.sqrt(7.0); r2 = r*r; r3 = r2*r
            poly = 1.0 + a*r + (14.0/5.0)*r2 + (7.0*a/15.0)*r3
            return amplitude * poly * jnp.exp(-a*r)
        else:
            raise ValueError("Supported ν: 3/2, 5/2, 7/2 only.")

    def __call__(self, x, y, domain):

        lengthscale, amplitude = self.get_transformed_params()

        r = domain.pairwise_geometry(x, y) / lengthscale
        nu = self.params.nu
        if nu == 1.5:
            a = jnp.sqrt(3.0); poly = 1.0 + a*r
            return amplitude * poly * jnp.exp(-a*r)
        elif nu == 2.5:
            a = jnp.sqrt(5.0); poly = 1.0 + a*r + (5.0/3.0)*r*r
            return amplitude * poly * jnp.exp(-a*r)
        elif nu == 3.5:  # 7/2
            a = jnp.sqrt(7.0); r2 = r*r; r3 = r2*r
            poly = 1.0 + a*r + (14.0/5.0)*r2 + (7.0*a/15.0)*r3
            return amplitude * poly * jnp.exp(-a*r)
        else:
            raise ValueError("Supported ν: 3/2, 5/2, 7/2 only.")


    def get_transformed_params(self):
        return (self.lengthscale_transform.forward(self.params.lengthscale),
                self.amplitude_transform.forward(self.params.amplitude))

    # pytree plumbing
    def tree_flatten(self):
        children = (self.params,)
        aux = (self.lengthscale_transform, self.amplitude_transform, self.left_shape, self.right_shape)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        ell_t, amp_t, lsh, rsh = aux
        (params,) = children
        return cls(params, ell_t, amp_t, lsh, rsh)

    @classmethod
    def axes(cls, params_axis: MaternParams):
        obj = object.__new__(cls)
        obj.params = params_axis
        obj.lengthscale_transform = cls.__dict__.get("lengthscale_transform", None) or Positive()
        obj.amplitude_transform = cls.__dict__.get("amplitude_transform", None) or Positive()
        obj.left_shape = ()
        obj.right_shape = ()
        return obj
