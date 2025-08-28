from __future__ import annotations
from typing import Protocol, Optional, Callable
from dataclasses import dataclass, field
from jax import Array
import jax
import jax.numpy as jnp 

from gaussed.likelihoods.gaussian_noise import NoiseSpec, Noiseless
from gaussed.gp.base import GP, PosteriorGP
from gaussed.linops import LinearOp, SumOp, AsLinearOp
from gaussed.types import LinearLike

@jax.tree_util.register_pytree_node_class
@dataclass
class GaussianLikelihood:
    """Gaussian observation model, parameterised by a NoiseSpec (operator-first)."""
    noise: NoiseSpec = field(default_factory=Noiseless)


    def add_to_gram(self, K_FF: LinearLike) -> LinearOp:
        K_FF = AsLinearOp(K_FF)
        n = K_FF.shape[0]
        return SumOp(K_FF, self.noise.as_op(n, K_FF.dtype))

    def tree_flatten(self):
        return (self.noise,), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        (noise,) = children
        return cls(noise)

    @classmethod
    def axes(cls, noise_axis):
        obj = object.__new__(cls)
        obj.noise = noise_axis
        return obj
