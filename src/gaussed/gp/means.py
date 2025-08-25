from dataclasses import dataclass
import jax, jax.numpy as jnp
from jax import Array
from typing import Tuple, Callable, Protocol

class MeanFunction(Protocol):
    def __call__(self, X): ...  # → (n,)