from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Tuple, Optional, Any, Callable
from jax import Array
import jax
import jax.numpy as jnp

from gaussed.gp.gp_ops.base import KernelSpec, FunSpec, OpContext
from gaussed.gp.gp_ops.probe import Probe, ProbeStack
from gaussed.types import LinearLike, ProbeLike
from gaussed.linops.constructors import LinOpConstructor

# ----- CovRep protocol: face for all reps --------------------------------
class CovRep(Protocol):

    @property
    def kernel_spec(self) -> KernelSpec: ...
    @property
    def mean_spec(self) -> FunSpec: ...

    @property
    def linop_constructor(self) -> LinOpConstructor: ...

    def gram(self, F: ProbeStack, G: ProbeStack, ctx: OpContext) -> LinearLike:
        """Return K_{FG} as Array (dense) or LinearOp."""
        ...

    def mean(self, F: ProbeStack, ctx: OpContext) -> LinearLike:
        """Return F[mean]."""
        ...
