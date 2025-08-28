from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Callable, Protocol, Any, Generic, TypeVar
import jax, jax.numpy as jnp
from jax import Array
from jax import tree_util as jtu

from gaussed.domains.base import Domain
from gaussed.gp.gp_ops.base import KernelSpec

# ========== Base Kernel (supports Tensor kernels) ==========
class Kernel(Protocol):
    left_shape: Tuple[int, ...]
    right_shape: Tuple[int, ...]

    def __call__(self, x: Array, y: Array, domain: Domain) -> Array: ...



@jax.tree_util.register_pytree_node_class
@dataclass
class KernelFunc:
    # (x,y,domain, params) -> left+right array
    fn: Callable[[Array, Array, Domain, Any], Array]

    params: Any # pytree (assumed to have correct constraints)
    left_shape: Tuple[int, ...]
    right_shape: Tuple[int, ...]

    def __call__(self, x: Array, y: Array, domain: Domain) -> Array:
        return self.fn(x, y, domain, self.params)

    # pytree plumbing
    def tree_flatten(self):
        children = (self.params,)
        aux = (self.fn, self.left_shape, self.right_shape)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        fn, lsh, rsh = aux
        (params,) = children
        return cls(fn, params, lsh, rsh)

