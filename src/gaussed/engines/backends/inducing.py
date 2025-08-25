from dataclasses import dataclass
from jax import Array

from gaussed.engines.linops import LinearOp, DenseOp, SumOp
from gaussed.engines.backends.solvers.base import Solver
from gaussed.gp.gp_ops import Probe, Eval
from gaussed.gp.base import PosteriorGP

@dataclass
class InducingBackend:
    solver: Solver
    Z: Array           # inducing inputs (m x d)
    approx: str = "vfe"  # or "fitc", "sgi", etc.

    def condition(self, gp, F: Probe, y: Array, lik) -> PosteriorGP:
        # Build K_FU, K_UU, etc. (can be dense or ops)
        K_FU = gp.K(F, Eval(self.Z))                 # (n, m)
        K_UU = gp.K(Eval(self.Z), Eval(self.Z))      # (m, m)

