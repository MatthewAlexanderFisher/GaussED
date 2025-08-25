from __future__ import annotations
from typing import Protocol, Optional, Callable
from dataclasses import dataclass
from jax import Array
import jax

from gaussed.gp.base import GP
from gaussed.gp.gp_ops.base import Operator, Probe
from gaussed.engines.linops.base import LinearOperator
from gaussed.gp.backends.noise import NoiseSpec

# Base Conditioner
class Conditioner(Protocol):
    """Holds factorisation/solver state of (K_FF + Σ)."""
    def solve(self, rhs: Array) -> Array: ...
    def logdet(self) -> Array: ...
    def solve_blocks(self, K_FQ: Array) -> Array:
        """Return (K_FF+Σ)^{-1} K_FQ; default uses batched solve."""
        return jax.vmap(self.solve, in_axes=1, out_axes=1)(K_FQ)
    
# Base Backend
class Backend(Protocol):
    name: str
    def condition(self, K_FF: Array, noise: "NoiseSpec") -> Conditioner: ...
