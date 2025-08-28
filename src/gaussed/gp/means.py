from dataclasses import dataclass
import jax, jax.numpy as jnp
from jax import Array
from typing import Tuple, Callable, Protocol

from gaussed.domains.base import Domain 

class MeanFun(Protocol):
    def __call__(self, X: Array, domain: Domain) -> Array: ...  # → (n,)

class ZeroMeanFun:

    def __call__(self, X: Array, domain: Domain) -> Array:
        # X of shape (batch_shape, event_shape)
        return jnp.zeros(X.shape[0])
