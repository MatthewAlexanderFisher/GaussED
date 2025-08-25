from dataclasses import dataclass
import jax, jax.numpy as jnp
from jax import Array
from typing import Tuple, Callable, Protocol


class Kernel(Protocol):
    def __call__(self, X, Xp): ...  # → (n, m)