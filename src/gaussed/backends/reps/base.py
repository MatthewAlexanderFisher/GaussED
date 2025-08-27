from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Tuple, Optional
from jax import Array
import jax.numpy as jnp

from gaussed.gp.gp_ops.base import KernelSpec, FunSpec, OpContext, Probe
from gaussed.types import LinearLike

# ----- CovRep protocol: face for all reps --------------------------------
class CovRep(Protocol):
    kernel_spec: KernelSpec
    mean_spec: FunSpec

    def gram(self, F: Probe, G: Probe, ctx: OpContext) -> LinearLike:
        """Return K_{FG} as Array (dense) or LinearOp."""
        ...

    def cross(self, F: Probe, G: Probe, ctx: OpContext) -> Array:
        """Return K_{FG} as dense Array (for prediction)."""
        ...

    def mean(self, F: Probe) -> Array:
        """Return F[mean]."""
        ...
