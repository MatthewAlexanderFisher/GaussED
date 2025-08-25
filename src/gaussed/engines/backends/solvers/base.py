from dataclasses import dataclass
from typing import Protocol, Optional
import jax.numpy as jnp
from jax import Array
from gaussed.engines.linops import LinearOp, DenseOp

class Factor(Protocol):
    """Factorisation/solver state for A (SPD)."""
    def solve(self, rhs: Array) -> Array: ...
    def solve_blocks(self, B: Array) -> Array: ...
    def logdet(self) -> Optional[Array]: ...      # None if unavailable
    def dtype(self) -> Array: ...                           # convenience

class Solver(Protocol):
    """Turn A (dense or LinearOp) into a Factor."""
    def factor(self, A: Array | LinearOp) -> Factor: ...

def as_linear_op(A: Array | LinearOp) -> LinearOp:
    return A if isinstance(A, LinearOp) else DenseOp(A)
