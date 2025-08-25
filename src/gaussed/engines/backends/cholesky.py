from __future__ import annotations
from typing import Protocol, Optional, Callable
from dataclasses import dataclass
from jax import Array
import jax
import jax.numpy as jnp

from gaussed.gp.base import GP
from gaussed.gp.gp_ops.base import Operator, Probe
from gaussed.engines.linops import LinearOp
from gaussed.engines.likelihood.gaussian_noise import NoiseSpec, Noiseless
from gaussed.engines.backends.base import Conditioner, Backend

@dataclass
class CholeskyState(Conditioner):
    L: Array  # lower Cholesky of (K_FF + Σ)

    def solve(self, rhs: Array) -> Array:
        return jax.scipy.linalg.cho_solve((self.L, True), rhs)

    def logdet(self) -> Array:
        return 2.0 * jnp.sum(jnp.log(jnp.diag(self.L)))

@dataclass
class CholeskyBackend(Backend):
    def condition(self, K_FF: Array, noise: NoiseSpec) -> Conditioner:
        Sigma = noise.as_matrix(K_FF.shape[0], K_FF.dtype)  # implement for each NoiseSpec
        A = K_FF + Sigma
        L = jnp.linalg.cholesky(A + 1e-6*jnp.eye(A.shape[0], A.dtype))
        return CholeskyState(L)
