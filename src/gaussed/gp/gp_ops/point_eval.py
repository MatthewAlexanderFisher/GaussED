from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Optional, Tuple, Callable
from jax import Array
import jax.numpy as jnp
import jax

from gaussed.gp.gp_ops.base import Functional, FunSpec, KernelSpec, OpContext

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Eval:
    X: Array  # (n_x, d)

    # unary: realise g at X
    def __call__(self, g: FunSpec, ctx: OpContext) -> Array:
        G = g.eval(self.X)
        return G if G.ndim == 2 else G[:, None]

    # left kernel map: Y ↦ k0(X, Y)  -> (n_x, q)
    def left_kernel_map(self, ks: KernelSpec, ctx: OpContext) -> Callable[[Array], Array]:
        X = self.X
        return lambda Y: ks.k0(X, Y)

    # right reduction: evaluate F at Y=X_right -> (n_L, n_R) with n_R = n_x
    def right_reduce(self, F_of_Y: Callable[[Array], Array], ctx: OpContext) -> Array:
        return F_of_Y(self.X)

    # pytree plumbing
    def tree_flatten(self): return ((self.X,), ())
    @classmethod
    def tree_unflatten(cls, aux, ch): (X,) = ch; return cls(X)

    def pair(self, other: Functional, ks: KernelSpec, ctx: OpContext) -> Array:
        return other.right_reduce(self.left_kernel_map(ks, ctx), ctx)


