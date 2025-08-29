from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Callable, Any, Protocol
import jax
import jax.numpy as jnp
from jax import tree_util as jtu
from jax import Array

class Domain(Protocol):
    @property
    def event_shape(self) -> Tuple[int, ...]: ...
    def project(self, x: Array) -> Array: ...
    def pairwise_geometry(self, x: Array, y: Array) -> Array: ...
    def default_nodes(self, n: int) -> Optional[Array]: ...

