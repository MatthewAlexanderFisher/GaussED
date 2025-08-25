from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Callable, Any
import jax
import jax.numpy as jnp
from jax import tree_util as jtu
from jax import Array

@jax.tree_util.register_pytree_node_class
@dataclass(init=False)
class Codomain:
    event_shape: Tuple[int, ...]
    coregionalisation: Optional[Array]

    def __init__(self, event_shape: Tuple[int, ...], coregionalisation: Optional[Array] = None):
        self.event_shape = tuple(event_shape)
        self.coregionalisation = None if coregionalisation is None else jnp.asarray(coregionalisation)

    def lift_cov(self, Kxy: Array) -> Array:
        C = self.coregionalisation
        return Kxy if C is None else Kxy[..., None, None] * C[None, None, :, :]

    def tree_flatten(self):
        # C can be None → keep a flag in aux
        children = tuple([] if self.coregionalisation is None else [self.coregionalisation])
        aux = (self.event_shape, self.coregionalisation is None)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        ev, none_flag = aux
        C = None if none_flag else children[0]
        return cls(ev, C)

    @classmethod
    def axes(cls, ev_axis, C_axis):
        obj = object.__new__(cls); obj.event_shape = ev_axis; obj.coregionalisation = C_axis
        return obj
