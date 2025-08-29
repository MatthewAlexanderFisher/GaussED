from __future__ import annotations
from typing import Protocol, Callable, Optional, Tuple
from dataclasses import dataclass
import jax, jax.numpy as jnp
from jax import Array

from gaussed.gp.gp_ops.probe import Probe
from gaussed.types import ProbeLike
from gaussed.gp.base import GP, PosteriorGP
from gaussed.linops.linop import LinearOp

class Likelihood(Protocol):
    def op_for(self, F: Probe, dtype) -> LinearOp: ...
    def gram_op(self, K_op: LinearOp, F: Probe, dtype) -> LinearOp: ...

    # Dense Convenience handlers
    def Sigma_for(self, F: Probe, dtype) -> Array: ...
    def add_to_gram(self, K_FF: Array, F: Probe) -> Array: ...

    def laplace_or_ep_condition(self, gp: GP, F: ProbeLike, y: Array) -> PosteriorGP:
        ...