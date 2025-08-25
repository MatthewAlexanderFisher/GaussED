from __future__ import annotations
from typing import Protocol, Optional, Tuple, Callable
from jax import Array
import jax.numpy as jnp
import jax

from gaussed.gp.kernels.base import Kernel
from gaussed.gp.means import MeanFunction
from gaussed.domains.base import Domain
from gaussed.gp.gp_ops.base import Operator, Probe


class Eval(Probe):
    def __init__(self, X: Array): 
        self.X = jnp.asarray(X)

    def n(self): 
        return self.X.shape[0]

    def mean(self, mean_fn: MeanFunction) -> Array: 
        return mean_fn(self.X)

    def K_with(self, other: Probe, kernel: Kernel, domain: Domain) -> Array:
        # Vectorised K(Eval(X), other)
        return other._K_as_rhs_from_eval(self.X, kernel, domain)

    # used when we're on the "left" (x-side)
    def _K_as_lhs_to_eval(self, Y, kernel, domain) -> Array:
        # K(Eval(X), Eval(Y)) = k(X,Y)
        def kxy(x): return jax.vmap(lambda y: kernel(x, y, domain))(Y)
        return jax.vmap(kxy)(self.X)

    # when we're on the "right" (y-side), other will call this
    def _K_as_rhs_from_eval(self, X, kernel, domain) -> Array:
        # symmetric to _K_as_lhs_to_eval
        def kxy(x): return jax.vmap(lambda y: kernel(x, y, domain))(self.X)
        return jax.vmap(kxy)(X).swapaxes(0,1)   # shape (len(X), len(self.X))
