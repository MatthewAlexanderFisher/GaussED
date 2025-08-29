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
    left_shape: Tuple[int, ...] = field(default_factory=lambda: (1,))
    right_shape: Tuple[int, ...] = field(default_factory=lambda: (1,))


    # scalar pair (x: (input_shape,), y: (input_shape,)) -> (left_shape, right_shape)
    def pair(self, x: Array, y: Array, domain: "Domain") -> Array:
        lengthscale, amplitude = self.get_transformed_params()   # ℓ > 0, σ² > 0
        # reshape ell for ARD to broadcast over input dims if needed
        in_shape = tuple(domain.input_shape)
        if lengthscale.ndim != 0:
            lengthscale = lengthscale.reshape(in_shape)
        # reuse batched path for correctness
        K = self.__call__(x.reshape((1, *in_shape)),
                          y.reshape((1, *in_shape)),
                          domain)  # (1,1,*L,*R)
        return K[0, 0, ...]         # (*L,*R)

    # Batched kernel: (n_f, *in) × (n_g, *in) -> (n_f, n_g, *L, *R)
    def __call__(self, x: Array, y: Array, domain: "Domain") -> Array:
        in_shape = tuple(domain.input_shape)
        x = domain.ensure_inputs(x)  # (n_f, *in)
        y = domain.ensure_inputs(y)  # (n_g, *in)
        n_f, n_g = x.shape[0], y.shape[0]

        lengthscale, amplitude = self.get_transformed_params()   # ℓ > 0, σ² > 0
        # ARD or scalar: broadcast by reshaping ell to (1,*in) if needed
        if lengthscale.ndim == 0:
            x_scaled, y_scaled = x / lengthscale, y / lengthscale
        else:
            lengthscale_b = lengthscale.reshape((1, *in_shape))
            x_scaled, y_scaled = x / lengthscale_b, y / lengthscale_b

        # pairwise squared distance -> (n_f, n_g)
        d2 = domain.squared_pairwise_geometry(x_scaled, y_scaled)

        K_base = amplitude * jnp.exp(-0.5 * d2)  # (n_f, n_g)

        # Base RBF is scalar-valued: enforce that |L|=|R|=1 and pack to (n_f,n_g,*L,*R)
        L, R = self.left_shape, self.right_shape
        K = K_base.reshape((n_f, n_g, *L, *R))  # adds the (1,1) tail in scalar case
        return K

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
