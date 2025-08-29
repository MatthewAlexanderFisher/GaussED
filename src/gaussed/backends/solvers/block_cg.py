from __future__ import annotations
from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple, Dict
import jax.numpy as jnp
from jax import Array
import jax

from gaussed.backends.solvers.linear_solver import LinearSolver, SolveFn, LinearSolverState, SqrtFn, LogdetFn
from gaussed.linops.linop import LinearOp, IdentityOp, AsLinearOp
from gaussed.types import LinearLike

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class BlockCGCache:
    x0: Optional[Array] = None  # warm start (n,) or (n,k)

    def tree_flatten(self):
        return (self.x0,), ()
    @classmethod
    def tree_unflatten(cls, aux, children):
        (x0,) = children
        return cls(x0)
