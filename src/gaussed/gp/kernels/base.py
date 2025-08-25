from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Callable, Protocol
import jax, jax.numpy as jnp
from jax import Array
from jax import tree_util as jtu

from gaussed.domains.base import Domain

class Kernel(Protocol):
    def __call__(self, x: Array, y: Array, domain: Domain) -> Array: ...  # → (n, m)
