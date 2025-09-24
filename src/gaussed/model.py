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
from gaussed.linops.linop import LinearOp, AsLinearOp, materialise_dense
from gaussed.utils.shape_helpers import pack_vec

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GPModel:
    gp: GP
    likelihood: "GaussianLikelihood" = field(
        default_factory=lambda: GaussianLikelihood()
    )

    def condition(self, F: ProbeLike, y: Array, *, backend: Optional["Backend"]=None) -> "PosteriorGP":
        be  = backend or self.gp.backend
        ks_or_phi, ms, ctx = be.make_specs(self)
        rep = be.make_rep(self)
        Fst = as_stack(F)

        # mean at training probes (raw): (nF, *out_shape)
        mF_raw  = materialise_dense(rep.mean(Fst, ctx))
        out_shape = tuple(mF_raw.shape[1:])
        nF = mF_raw.shape[0]

        # coerce y to raw shape, then pack RHS as a **column**
        y_raw = y if (y.ndim == 1 and not out_shape) else y.reshape((nF, *out_shape))
        rhs   = pack_vec(y_raw - mF_raw, out_shape)                 # (nF*L, 1)

        # training Gram via your constructor (already flattened)
        K_FF  = rep.linop_constructor(ks_or_phi, Fst, Fst, ctx)  # (nF*L, nF*L)
        A     = self.likelihood.add_to_gram(K_FF)
        solver = LinearSolver(A, be.solverfns, be.solver_state)

        # RHS, solver returns (nF*L, 1)
        alpha = materialise_dense(solver.solve(AsLinearOp(rhs)))

        return PosteriorGP(
            gp=self.gp, rep=rep, F=Fst, ctx=ctx, solver=solver,
            alpha=alpha, y=y_raw, mF=mF_raw
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