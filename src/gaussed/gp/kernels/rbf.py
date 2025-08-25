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
    ell_transform: Transform = field(default_factory=Positive)
    sigma2_transform: Transform = field(default_factory=Positive)

    def __call__(self, x: Array, y: Array, domain: Domain) -> Array:
        ell = self.ell_transform.forward(self.params.lengthscale)
        s2 = self.sigma2_transform.forward(self.params.amplitude)

        r = domain.pairwise_geometry(x, y) / (ell + 1e-12)
        return s2 * jnp.exp(-0.5 * r * r)

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
        ell = self.ell_transform.forward(self.params.lengthscale)   # ℓ > 0
        s2  = self.sigma2_transform.forward(self.params.amplitude)  # σ² > 0
        const = s2 * jnp.sqrt(2.0 * jnp.pi) * ell                   # σ² √(2π) ℓ
        return const * jnp.exp(-0.5 * (ell * omega) ** 2)

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
    