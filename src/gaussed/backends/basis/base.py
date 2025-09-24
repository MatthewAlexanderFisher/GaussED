from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Callable, Protocol, Optional

from jax import Array
import jax
import jax.numpy as jnp

from gaussed.types import ProbeLike
from gaussed.gp.gp_ops.base import KernelSpec, FunSpec

class Basis(Protocol):
    @property
    def kernel_spec(self) -> KernelSpec: ...
    @property
    def mean_spec(self) -> FunSpec: ...
    @property
    def phi_spec(self) -> FunSpec: ...
    @property
    def Lambda(self) -> Array: ...  # (m,), (m,m), or scalar

