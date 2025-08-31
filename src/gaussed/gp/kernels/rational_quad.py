from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Callable
import jax, jax.numpy as jnp
from jax import Array

from gaussed.domains.base import Domain
from gaussed.utils.constraints import Positive, Transform


# --- Rational Quadratic --------------------------------------------------------
@jax.tree_util.register_pytree_node_class
@dataclass
class RQParams:
    lengthscale: Array          # unconstrained
    amplitude: Array
    alpha: Array                # unconstrained -> transformed to (0, ∞)

    def tree_flatten(self):
        return (self.lengthscale, self.amplitude, self.alpha), None

    @classmethod
    def tree_unflatten(cls, aux, ch):
        return cls(*ch)

    @classmethod
    def axes(cls, ell_axis, s2_axis, alpha_axis):
        obj = object.__new__(cls)
        obj.lengthscale = ell_axis
        obj.amplitude   = s2_axis
        obj.alpha       = alpha_axis
        return obj


@jax.tree_util.register_pytree_node_class
@dataclass
class RQKernel:
    params: RQParams
    lengthscale_transform: Transform = field(default_factory=Positive)
    amplitude_transform: Transform   = field(default_factory=Positive)
    alpha_transform: Transform       = field(default_factory=Positive)

    # tensor-kernel interface (static; not part of pytree children)
    left_shape: Tuple[int, ...]  = field(default_factory=lambda: (1,))
    right_shape: Tuple[int, ...] = field(default_factory=lambda: (1,))

    # scalar pair (x: (input_shape,), y: (input_shape,)) -> (left_shape, right_shape)
    def pair(self, x: Array, y: Array, domain: "Domain") -> Array:
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

        lengthscale, amplitude, alpha = self.get_transformed_params()  # ℓ>0, σ²>0, α>0

        # ARD or scalar length-scale
        if lengthscale.ndim == 0:
            denom = (lengthscale + 1e-12)
            x_scaled, y_scaled = x / denom, y / denom
        else:
            lengthscale_b = lengthscale.reshape((1, *in_shape))
            x_scaled, y_scaled = x / (lengthscale_b + 1e-12), y / (lengthscale_b + 1e-12)

        # pairwise squared distance of scaled inputs -> (n_f, n_g)
        d2 = domain.squared_pairwise_geometry(x_scaled, y_scaled)

        # Rational Quadratic:
        # k = σ² * (1 + d2/(2α))^{-α}
        base = 1.0 + d2 / (2.0 * (alpha + 1e-12))
        K_base = amplitude * jnp.power(base, -alpha)

        # Scalar-valued kernel → pack to (n_f, n_g, *L, *R)
        L, R = self.left_shape, self.right_shape
        K = K_base.reshape((n_f, n_g, *L, *R))
        return K

    def get_transformed_params(self):
        return (
            self.lengthscale_transform.forward(self.params.lengthscale),
            self.amplitude_transform.forward(self.params.amplitude),
            self.alpha_transform.forward(self.params.alpha),
        )

    # pytree plumbing
    def tree_flatten(self):
        children = (self.params,)
        aux = (self.lengthscale_transform,
               self.amplitude_transform,
               self.alpha_transform,
               self.left_shape,
               self.right_shape)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        ell_t, s2_t, a_t, lsh, rsh = aux
        (params,) = children
        return cls(params, ell_t, s2_t, a_t, lsh, rsh)

    @classmethod
    def axes(cls, params_axis: RQParams):
        obj = object.__new__(cls)
        obj.params = params_axis
        obj.lengthscale_transform = cls.__dict__.get("lengthscale_transform", None) or Positive()
        obj.amplitude_transform   = cls.__dict__.get("amplitude_transform", None)   or Positive()
        obj.alpha_transform       = cls.__dict__.get("alpha_transform", None)       or Positive()
        obj.left_shape = ()
        obj.right_shape = ()
        return obj
