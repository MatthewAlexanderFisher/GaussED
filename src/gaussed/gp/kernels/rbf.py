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
    lengthscale: Array          # unconstrained
    amplitude: Array

    def tree_flatten(self): 
        return (self.lengthscale, self.amplitude), None
    
    @classmethod
    def tree_unflatten(cls, aux, ch): 
        return cls(*ch)

    @classmethod
    def axes(cls, ell_axis, s2_axis):
        obj = object.__new__(cls)
        obj.lengthscale = ell_axis
        obj.amplitude = s2_axis
        return obj

@jax.tree_util.register_pytree_node_class
@dataclass
class RBFKernel:
    params: RBFParams
    lengthscale_transform: Transform = field(default_factory=Positive)
    amplitude_transform: Transform = field(default_factory=Positive)

    # tensor-kernel interface (static; not part of pytree children)
    left_shape: Tuple[int, ...] = field(default_factory=tuple)
    right_shape: Tuple[int, ...] = field(default_factory=tuple)

    # scalar pair (x: (d,), y: (d,)) -> ()
    def pair(self, x: Array, y: Array, domain: "Domain") -> Array:
        lengthscale, amplitude = self.get_transformed_params()   # ℓ > 0, σ² > 0
        r = domain.pairwise_geometry(x, y) / (lengthscale + 1e-12)
        return amplitude * jnp.exp(-0.5 * r * r)


    def __call__(self, x: Array, y: Array, domain: Domain) -> Array:
        lengthscale, amplitude = self.get_transformed_params()   # ℓ > 0, σ² > 0
        r = domain.pairwise_geometry(x, y) / (lengthscale + 1e-12)
        return amplitude * jnp.exp(-0.5 * r * r)

    def spectral_density(self, omega: Array) -> Array:
        r"""
        Spectral density :math:`S(\omega)` of the 1D RBF kernel under the 
        angular-frequency convention.

        Fourier convention (angular frequency):

        .. math::

            S(\omega) = \int_{-\infty}^{\infty} k(\tau) e^{-i \omega \tau}\, d\tau,
            \qquad
            k(\tau) = \frac{1}{2\pi}\int_{-\infty}^{\infty} S(\omega) e^{i \omega \tau}\, d\omega.

        For the squared exponential (RBF) kernel

        .. math::

            k(\tau) = \sigma^{2} \exp\!\left(-\tfrac{\tau^{2}}{2\ell^{2}}\right),

        the spectral density is

        .. math::

            S(\omega) = \sigma^{2}\,\sqrt{2\pi}\,\ell \,
                        \exp\!\left(-\tfrac{1}{2}\ell^{2}\omega^{2}\right).

        Args:
            omega: Array of angular frequencies :math:`\omega` (radians per unit).

        Returns:
            Array broadcasting over ``omega`` and any batch dimensions of the parameters.
        """
        lengthscale, amplitude = self.get_transformed_params()   # ℓ > 0, σ² > 0
        const = amplitude * jnp.sqrt(2.0 * jnp.pi) * lengthscale  # σ² √(2π) ℓ
        return const * jnp.exp(-0.5 * (lengthscale * omega) ** 2)

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
        ell_t, s2_t, lsh, rsh = aux
        (params,) = children
        return cls(params, ell_t, s2_t, lsh, rsh)

    @classmethod
    def axes(cls, params_axis: RBFParams):
        obj = object.__new__(cls)
        obj.params = params_axis
        obj.lengthscale_transform = cls.__dict__.get("lengthscale_transform", None) or Positive()
        obj.amplitude_transform = cls.__dict__.get("amplitude_transform", None) or Positive()
        obj.left_shape = ()
        obj.right_shape = ()
        return obj
