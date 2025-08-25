from dataclasses import dataclass
from typing import Callable, Protocol
from jax import Array

from gaussed.gp.base import GP, PosteriorGP
from gaussed.gp.gp_ops import Probe
from gaussed.gp.base import PosteriorGP
from gaussed.engines.backends.solvers.base import Solver

@dataclass
class BasisBackend:
    solver: Solver
    features: Callable      # Φ: (Probe) -> (n, m) or LinearOp
    prior_w_precision: Array | float  # Λ^{-1} or scalar

    def condition(self, gp, F: Probe, y: Array, lik) -> PosteriorGP:
        # A = Φ Λ^{-1} Φ^T + Σ   (n x n, SPD)
        Phi = self.features(F)             # (n, m) or LinearOp (n×m)
        # Build A as an op or dense; factor with solver; continue as usual
        

@dataclass
class Basis(Protocol):
    def __call__(self, F: Probe) -> Array:
        ...