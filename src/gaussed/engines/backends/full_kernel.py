from dataclasses import dataclass
from jax import Array
from gaussed.engines.linops import LinearOp, DenseOp, SumOp
from gaussed.gp.base import GP, PosteriorGP
from gaussed.gp.gp_ops.base import Probe
from gaussed.engines.likelihoods.gaussian import GaussianLikelihood
from gaussed.engines.backends.solvers.base import Solver

@dataclass
class FullKernelBackend:
    solver: Solver

    def condition(self, gp: GP, F: Probe, y: Array, lik: GaussianLikelihood) -> PosteriorGP:
        # Build K and Σ in the form the solver prefers
        K = gp.K(F, F)                      # dense for now; you can make a kernel op, too
        K_op = DenseOp(K)
        Sigma_op = lik.op_for(F, dtype=K.dtype)
        A_op = SumOp(K_op, Sigma_op)
        # Factor and solve
        fac = self.solver.factor(A_op)      # works for Cholesky (densifies) or PCG (matvec)
        mu_F = F.mean(gp.mean)
        alpha = fac.solve(y - mu_F)
        # Build a PosteriorGP-like state that carries the factor for predictions
        return PosteriorGP(gp, F, fac, alpha)
