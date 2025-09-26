from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple
import jax, jax.numpy as jnp
from jax import Array

from gaussed.domains.base import Domain
from gaussed.utils.constraints import Positive, Transform


# --- Cauchy -------------------------------------------------------------------
@jax.tree_util.register_pytree_node_class
@dataclass
class CauchyParams:
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
class CauchyKernel:
    params: CauchyParams
    lengthscale_transform: Transform = field(default_factory=Positive)
    amplitude_transform: Transform = field(default_factory=Positive)

    # tensor-kernel interface (static; not part of pytree children)
    left_shape: Tuple[int, ...] = field(default_factory=lambda: (1,))
    right_shape: Tuple[int, ...] = field(default_factory=lambda: (1,))

    # scalar pair (x: (input_shape,), y: (input_shape,)) -> (left_shape, right_shape)
    def pair(self, x: Array, y: Array, domain: "Domain") -> Array:
        # Reuse batched path for correctness/broadcasting
        in_shape = tuple(domain.input_shape)
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

        # ARD or scalar: scale inputs by ℓ
        if lengthscale.ndim == 0:
            x_scaled, y_scaled = x / lengthscale, y / lengthscale
        else:
            lengthscale_b = lengthscale.reshape((1, *in_shape))
            x_scaled, y_scaled = x / lengthscale_b, y / lengthscale_b

        # pairwise squared (scaled) distance -> (n_f, n_g)
        d2 = domain.squared_pairwise_geometry(x_scaled, y_scaled)

        # Cauchy kernel: k(r) = σ² / (1 + r²), with r = ||(x - y)/ℓ||
        K_base = amplitude / (1.0 + d2)  # (n_f, n_g)

        # Pack to (n_f, n_g, *L, *R)
        L, R = self.left_shape, self.right_shape
        K = K_base.reshape((n_f, n_g, *L, *R))
        return K

    def spectral_density_1d(self, omega: Array) -> Array:
        r"""
        1D spectral density under the angular-frequency convention:

            S(ω) = ∫ k(τ) e^{-i ω τ} dτ,   k(τ) = (1/2π) ∫ S(ω) e^{i ω τ} dω.

        For the Cauchy kernel
            k(τ) = σ² / (1 + (τ/ℓ)²) = σ² ℓ² / (τ² + ℓ²),

        using ∫_{-∞}^{∞} e^{-i ω τ} (1/(τ²+a²)) dτ = π/a · e^{-a|ω|},

        we obtain:
            S(ω) = σ² · π ℓ · e^{-ℓ |ω|}.
        """
        lengthscale, amplitude = self.get_transformed_params()   # ℓ > 0, σ² > 0
        return amplitude * jnp.pi * lengthscale * jnp.exp(-lengthscale * jnp.abs(omega))

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
    def axes(cls, params_axis: CauchyParams):
        obj = object.__new__(cls)
        obj.params = params_axis
        obj.lengthscale_transform = cls.__dict__.get("lengthscale_transform", None) or Positive()
        obj.amplitude_transform = cls.__dict__.get("amplitude_transform", None) or Positive()
        obj.left_shape = ()
        obj.right_shape = ()
        return obj
