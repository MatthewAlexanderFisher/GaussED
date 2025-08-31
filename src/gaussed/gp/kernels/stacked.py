from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Callable, Any
import jax, jax.numpy as jnp
from jax import Array

from gaussed.domains.base import Domain
from gaussed.utils.constraints import Positive, Transform


@jax.tree_util.register_pytree_node_class
@dataclass
class StackedKernel:
    kernels: Tuple[Any, ...]                    # each returns (n_f, n_g, 1, 1)
    enable_checks: bool = True

    # derived/static (not pytree children)
    left_shape: Tuple[int, ...]  = field(init=False)
    right_shape: Tuple[int, ...] = field(init=False)

    def __post_init__(self):
        m = len(self.kernels)
        object.__setattr__(self, "left_shape",  (m,))
        object.__setattr__(self, "right_shape", (m,))
        if self.enable_checks:
            for k in self.kernels:
                lsh = tuple(getattr(k, "left_shape",  (1,)))
                rsh = tuple(getattr(k, "right_shape", (1,)))
                if lsh != (1,) or rsh != (1,):
                    raise ValueError(
                        "StackedKernel expects each child kernel to be scalar-valued "
                        f"(left_shape=right_shape=(1,)), got {lsh} and {rsh}."
                    )

    # pair: (input_shape,), (input_shape,) -> (m, m)
    def pair(self, x: Array, y: Array, domain: "Domain") -> Array:
        in_shape = tuple(domain.input_shape)
        K = self.__call__(x.reshape((1, *in_shape)),
                          y.reshape((1, *in_shape)),
                          domain)               # (1,1,m,m)
        return K[0, 0, ...]                     # (m, m)

    # call: (n_f,*in) × (n_g,*in) -> (n_f, n_g, m, m)
    def __call__(self, x: Array, y: Array, domain: "Domain") -> Array:
        x = domain.ensure_inputs(x)             # (n_f, *in)
        y = domain.ensure_inputs(y)             # (n_g, *in)
        n_f, n_g = x.shape[0], y.shape[0]
        m = len(self.kernels)

        # Evaluate each scalar kernel → stack → diagonalise
        K_list = []
        for k in self.kernels:
            Kij = k(x, y, domain)               # (n_f, n_g, 1, 1)
            K_list.append(Kij.reshape(n_f, n_g))  # (n_f, n_g)
        K_diag = jnp.stack(K_list, axis=-1)     # (n_f, n_g, m)

        I = jnp.eye(m)                          # (m, m)
        K = jnp.einsum("fgk,ik,jk->fgij", K_diag, I, I)  # (n_f, n_g, m, m)
        return K  # already (n_f, n_g, *left_shape, *right_shape)

    # pytree plumbing: shapes are derived; don’t store them in aux
    def tree_flatten(self):
        return (self.kernels,), (self.enable_checks,)

    @classmethod
    def tree_unflatten(cls, aux, children):
        (enable_checks,) = aux
        (kernels,) = children
        return cls(tuple(kernels), enable_checks=enable_checks)

    @classmethod
    def axes(cls, kernels_axis: Tuple[Any, ...], enable_checks: bool = True):
        # Shapes derive from len(kernels_axis) at construction time
        obj = object.__new__(cls)
        obj.kernels = kernels_axis
        obj.enable_checks = enable_checks
        # Derive shapes now (static):
        m = len(kernels_axis)
        obj.left_shape = (m,)
        obj.right_shape = (m,)
        return obj
