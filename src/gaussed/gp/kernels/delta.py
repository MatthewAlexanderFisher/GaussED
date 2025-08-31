from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Callable
import jax, jax.numpy as jnp
from jax import Array

from gaussed.domains.base import Domain
from gaussed.utils.constraints import Positive, Transform

@jax.tree_util.register_pytree_node_class
@dataclass
class DeltaParams:
    amplitude: Array  # unconstrained

    def tree_flatten(self):
        return (self.amplitude,), ()

    @classmethod
    def tree_unflatten(cls, aux, children):
        (amp,) = children
        return cls(amp)

@jax.tree_util.register_pytree_node_class
@dataclass
class DeltaKernel:
    """k(i,j) = α · 1{i=j}. Intended to be used only on IndexDomain. Left/right shapes are scalar."""
    params: DeltaParams
    amp_transform: Positive = field(default_factory=Positive)

    left_shape: Tuple[int, ...]  = field(default_factory=lambda: (1,))
    right_shape: Tuple[int, ...] = field(default_factory=lambda: (1,))

    def pair(self, x: Array, y: Array, domain: Domain) -> Array:
        in_shape = tuple(domain.input_shape)
        K = self.__call__(x.reshape((1, *in_shape)),
                          y.reshape((1, *in_shape)),
                          domain)  # (1,1,*L,*R)
        return K[0, 0, ...]  # (*L,*R)

    # (nF, nG, *L, *R) = (nF, nG)
    def __call__(self, x, y, domain):
        alpha = self.amp_transform.forward(self.params.amplitude)
        d = domain.pairwise_geometry(x, y)                    # (nF, nG)
        K = alpha * (d == 0).astype(d.dtype)                      # (nF, nG)
        return K[..., None, None]                             # (nF, nG, 1, 1)

    def tree_flatten(self):
        return (self.params,), (self.amp_transform,)

    @classmethod
    def tree_unflatten(cls, aux, children):
        (amp_t,) = aux
        (params,) = children
        return cls(params, amp_t)

    @classmethod
    def axes(cls, params_axis: DeltaParams):
        obj = object.__new__(cls)
        obj.params = params_axis
        obj.amp_transform = Positive()
        return obj
