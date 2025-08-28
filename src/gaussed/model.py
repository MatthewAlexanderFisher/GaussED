from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Callable, Any, Union
import jax
import jax.numpy as jnp
from jax import Array

from gaussed.gp.base import GP, PosteriorGP
from gaussed.backends.base import Backend
from gaussed.likelihoods.base import Likelihood
from gaussed.likelihoods.gaussian import GaussianLikelihood
from gaussed.types import ProbeLike
from gaussed.gp.gp_ops.probe import Probe, ProbeStack, as_stack

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GPModel:
    gp: GP
    likelihood: GaussianLikelihood = GaussianLikelihood()

    def condition(self, F: ProbeLike, y: Array, *, backend: Optional[Backend]=None) -> "PosteriorGP":
        be = backend or self.gp.backend
        ks, ms, ctx = be.make_specs(self)
        rep = be.make_rep(self)

        Fst = as_stack(F)

        # Build training blocks
        mF = rep.mean(Fst, ctx)                 # (n,)
        K_FF = rep.gram(Fst, Fst, ctx)          # (n,n) dense here (Exact)
        A = self.likelihood.add_to_gram(K_FF)   # (n,n)

        solver = be.solver.rebind(A)

        r = y - mF                               # (n,)
        alpha = solver.solve(r)                  # (n,)

        return PosteriorGP(
            gp=self.gp,
            rep=rep,
            F=Fst,
            ctx=ctx,
            solver=solver,
            alpha=alpha,
            y=y,
            mF=mF
        )
