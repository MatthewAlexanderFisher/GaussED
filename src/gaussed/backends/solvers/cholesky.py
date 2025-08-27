from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jax import Array

from gaussed.backends.solvers.base import Solver, Factor
from gaussed.linops import LinearOp, AsLinearOp

@dataclass
class CholFactor(Factor):
    L: Array
    _dtype: Array

    def solve(self, u: Array) -> Array:
        return jax.scipy.linalg.cho_solve((self.L, True), u)
    
    def solve_blocks(self, B: Array) -> Array:
        return jax.scipy.linalg.cho_solve((self.L, True), B)
    
    def logdet(self) -> Array:
        return 2.0 * jnp.sum(jnp.log(jnp.diag(self.L)))
    
    def dtype(self) -> Array: 
        return self._dtype

@dataclass
class CholeskySolver(Solver):
    jitter: float = 1e-6
    def factor(self, A: Array | LinearOp) -> Factor:
        # If given an op, densify once here
        op = AsLinearOp(A)
        M = op.to_dense()
        n = M.shape[0]
        L = jnp.linalg.cholesky(M + self.jitter * jnp.eye(n, dtype=M.dtype))
        return CholFactor(L, M.dtype)
