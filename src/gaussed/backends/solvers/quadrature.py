from dataclasses import dataclass
from typing import Protocol, Optional, Callable, Any, Tuple
import jax
import jax.numpy as jnp
from jax import Array
from gaussed.linops import LinearOp, DenseOp, AsLinearOp

from gaussed.domains.base import Domain

# TODO: implement get nodes/weights for quadrature routines (domain dependent...)