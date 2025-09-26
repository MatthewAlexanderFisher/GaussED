from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Callable
import jax, jax.numpy as jnp
from jax import Array
from jax.scipy.special import gamma

from gaussed.domains.base import Domain
from gaussed.utils.constraints import Positive, Transform


# --- IMQ / Inverse-Multiquadric ------------------------------------------------
@jax.tree_util.register_pytree_node_class
@dataclass
class IMQParams:
    lengthscale: Array          # unconstrained
    amplitude: Array
    beta: Array                 # unconstrained -> transformed to (0, ∞)

    def tree_flatten(self):
        return (self.lengthscale, self.amplitude, self.beta), None

    @classmethod
    def tree_unflatten(cls, aux, ch):
        return cls(*ch)

    @classmethod
    def axes(cls, ell_axis, s2_axis, beta_axis):
        obj = object.__new__(cls)
        obj.lengthscale = ell_axis
        obj.amplitude = s2_axis
        obj.beta = beta_axis
        return obj


@jax.tree_util.register_pytree_node_class
@dataclass
class IMQKernel:
    params: IMQParams
    lengthscale_transform: Transform = field(default_factory=Positive)
    amplitude_transform: Transform   = field(default_factory=Positive)
    beta_transform: Transform        = field(default_factory=Positive)

    # tensor-kernel interface (static; not part of pytree children)
    left_shape: Tuple[int, ...]  = field(default_factory=lambda: (1,))
    right_shape: Tuple[int, ...] = field(default_factory=lambda: (1,))

    # scalar pair (x: (input_shape,), y: (input_shape,)) -> (left_shape, right_shape)
    def pair(self, x: Array, y: Array, domain: "Domain") -> Array:
        # Reuse batched path for correctness and to keep shapes consistent
        in_shape = tuple(domain.input_shape)
        K = self.__call__(x.reshape((1, *in_shape)),
                          y.reshape((1, *in_shape)),
                          domain)  # (1,1,*L,*R)
        return K[0, 0, ...]  # (*L,*R)

    # Batched kernel: (n_f, *in) × (n_g, *in) -> (n_f, n_g, *L, *R)
    def __call__(self, x: Array, y: Array, domain: "Domain") -> Array:
        in_shape = tuple(domain.input_shape)
        x = domain.ensure_inputs(x)  # (n_f, *in)
        y = domain.ensure_inputs(y)  # (n_g, *in)
        n_f, n_g = x.shape[0], y.shape[0]

        lengthscale, amplitude, beta = self.get_transformed_params()  # ℓ>0, σ²>0, β>0

        # ARD or scalar: broadcast by reshaping ell to (1,*in) if needed
        if lengthscale.ndim == 0:
            x_scaled, y_scaled = x / (lengthscale + 1e-12), y / (lengthscale + 1e-12)
        else:
            lengthscale_b = lengthscale.reshape((1, *in_shape))
            x_scaled, y_scaled = x / (lengthscale_b + 1e-12), y / (lengthscale_b + 1e-12)

        # pairwise squared distance of scaled inputs -> (n_f, n_g)
        d2 = domain.squared_pairwise_geometry(x_scaled, y_scaled)

        # IMQ base kernel (heavier tails than RBF):
        # k = σ² * (1 + d2)^(-β)
        K_base = amplitude * jnp.power(1.0 + d2, -beta)

        # Base IMQ is scalar-valued: pack to (n_f, n_g, *L, *R)
        L, R = self.left_shape, self.right_shape
        K = K_base.reshape((n_f, n_g, *L, *R))
        return K


    def spectral_density_1d(self, omega: jnp.ndarray, *, N: int = 1024) -> jnp.ndarray:
        r"""
        1D spectral density S(ω) for IMQ k(τ)=σ² (1+(τ/ℓ)²)^(-β), using the angular-frequency pair:
            S(ω) = ∫ k(τ) e^{-iωτ} dτ,   k(τ) = (1/2π) ∫ S(ω) e^{iωτ} dω.

        Quadrature form (τ = ℓ tan t):
            S(ω) = 2 σ² ℓ ∫_{0}^{π/2} cos^{2β-2}(t) · cos(ω ℓ tan t) dt.

        Args:
            omega: array of angular frequencies (rad / unit).
            N:     number of midpoint samples over (0, π/2); increase for higher accuracy.

        Returns:
            S(ω) with the same shape/broadcasting as `omega`.
        """
        ell, sigma2, beta = self.get_transformed_params()  # ℓ>0, σ²>0, β>1/2 recommended

        w = jnp.abs(omega)

        # Exact value at ω=0:
        # S(0) = 2 σ² ∫_0^∞ (1+(τ/ℓ)^2)^(-β) dτ = σ² ℓ √π Γ(β-1/2)/Γ(β)
        S0 = sigma2 * ell * jnp.sqrt(jnp.pi) * gamma(beta - 0.5) / gamma(beta)

        # exact Cauchy case β=1
        def s_cauchy(w_):
            return sigma2 * jnp.pi * ell * jnp.exp(-ell * w_)
        use_cauchy = jnp.isclose(beta, 1.0)

        # Midpoint rule over t ∈ (0, π/2)
        # t_k = (k+1/2) * h,  k=0..N-1,  h = (π/2)/N
        h = 0.5 * jnp.pi / float(N)
        k = jnp.arange(N, dtype=ell.dtype)
        t = (k + 0.5) * h
        cos_t = jnp.cos(t)
        tan_t = jnp.tan(t)

        # Weight factor: cos^{2β-2}(t); note integrability requires β > 1/2
        weight = cos_t ** (2.0 * beta - 2.0)  # (N,)

        # Broadcast over omega: shape (..., N)
        z = (ell * w)[..., None]              # (..., 1)
        integrand = ell * weight[None, :] * jnp.cos(z * tan_t[None, :])  # (..., N)

        I = h * jnp.sum(integrand, axis=-1)   # (...,)

        S_num = 2.0 * sigma2 * I

        # Blend exact ω=0 value to remove removable singularity numerically
        S_num = jnp.where(w == 0, S0, S_num)

        # exact β=1 shortcut (closed form)
        return jnp.where(use_cauchy, s_cauchy(w), S_num)


    def get_transformed_params(self):
        return (
            self.lengthscale_transform.forward(self.params.lengthscale),
            self.amplitude_transform.forward(self.params.amplitude),
            self.beta_transform.forward(self.params.beta),
        )

    # pytree plumbing
    def tree_flatten(self):
        children = (self.params,)
        aux = (self.lengthscale_transform,
               self.amplitude_transform,
               self.beta_transform,
               self.left_shape,
               self.right_shape)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        ell_t, s2_t, b_t, lsh, rsh = aux
        (params,) = children
        return cls(params, ell_t, s2_t, b_t, lsh, rsh)

    @classmethod
    def axes(cls, params_axis: IMQParams):
        obj = object.__new__(cls)
        obj.params = params_axis
        obj.lengthscale_transform = cls.__dict__.get("lengthscale_transform", None) or Positive()
        obj.amplitude_transform   = cls.__dict__.get("amplitude_transform", None)   or Positive()
        obj.beta_transform        = cls.__dict__.get("beta_transform", None)        or Positive()
        obj.left_shape = ()
        obj.right_shape = ()
        return obj
