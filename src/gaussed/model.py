from __future__ import annotations
from dataclasses import dataclass, field
import jax
import jax.numpy as jnp
from jax import Array

from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from gaussed.backends.base import Backend


from gaussed.gp.base import GP, PosteriorGP
from gaussed.likelihoods.base import Likelihood
from gaussed.likelihoods.gaussian import GaussianLikelihood
from gaussed.types import ProbeLike
from gaussed.gp.gp_ops.probe import Probe, ProbeStack, as_stack
from gaussed.backends.solvers.linear_solver import LinearSolver

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GPModel:
    gp: GP
    likelihood: "GaussianLikelihood" = field(
        default_factory=lambda: GaussianLikelihood()
    )

    def condition(self, F: ProbeLike, y: Array, *, backend: Optional["Backend"]=None) -> "PosteriorGP":
        be = backend or self.gp.backend
        ks, ms, ctx = be.make_specs(self)
        rep = be.make_rep(self)

        Fst = as_stack(F)

        # Build training blocks
        mF = rep.mean(Fst, ctx)                 # (n,)
        K_FF = rep.gram(Fst, Fst, ctx)          # (n,n) dense here (Exact)
        A = self.likelihood.add_to_gram(K_FF)   # (n,n)

        solver = LinearSolver(A, be.solverfns, be.solver_state)

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

    # --- pytree plumbing ---
    def tree_flatten(self):
        # both are children so params/state trace through jit/scan
        children = (self.gp, self.likelihood)
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        gp, likelihood = children
        obj = object.__new__(cls)
        object.__setattr__(obj, "gp", gp)
        object.__setattr__(obj, "likelihood", likelihood)
        return obj

    @classmethod
    def axes(cls, gp_axes: "GP", lik_axes: "GaussianLikelihood"):
        obj = object.__new__(cls)
        object.__setattr__(obj, "gp", gp_axes)
        object.__setattr__(obj, "likelihood", lik_axes)
        return obj