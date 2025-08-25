from __future__ import annotations
from typing import Protocol, Optional, Callable, Tuple
from dataclasses import dataclass
from jax import Array
import jax
import jax.numpy as jnp

from gaussed.gp.base import GP
from gaussed.gp.gp_ops.base import Operator, Probe
from gaussed.engines.linops import LinearOp
from gaussed.engines.likelihood.gaussian_noise import NoiseSpec
from gaussed.engines.backends.base import Conditioner, Backend
from gaussed.engines.linops import LinearOp

@dataclass
class CGState(Conditioner):
    A: LinearOp
    tol: float = 1e-6
    maxit: int = 512
    def solve(self, rhs: Array) -> Array:
        # implement PCG or use jaxopt; left as stub
        x0 = jnp.zeros_like(rhs)
        return pcg(self.A.mv, rhs, x0, self.tol, self.maxit)
    def logdet(self) -> Array:
        # stochastic Lanczos trace estimator, or expose as NotImplemented
        raise NotImplementedError
