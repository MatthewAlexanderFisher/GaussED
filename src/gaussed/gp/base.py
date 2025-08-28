from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Callable, Any, Union
import jax
import jax.numpy as jnp
from jax import Array

from gaussed.backends.base import Backend
from gaussed.gp.kernels.base import Kernel
from gaussed.gp.means import MeanFun
from gaussed.domains.base import Domain
from gaussed.codomains.base import Codomain
from gaussed.gp.gp_ops.base import OpContext
from gaussed.gp.gp_ops.probe import Probe, ProbeStack, as_stack
from gaussed.backends.base import Backend
from gaussed.types import ProbeLike
from gaussed.gp.gp_ops.base import FunSpec, KernelSpec
from gaussed.utils.make_specs import make_fun_spec, make_kernel_spec
from gaussed.backends.reps.base import CovRep
from gaussed.backends.solvers.linear_solver import LinearSolver


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GP:
    domain: Domain
    codomain: Codomain
    mean: MeanFun      # your MeanFun object
    kernel: Kernel    # your Kernel object
    backend: Backend
    op_ctx: Optional[OpContext] = None

    # bridge to specs
    def mean_spec(self) -> FunSpec:
        return make_fun_spec(self.mean, self.domain)

    def kernel_spec(self) -> KernelSpec:
        return make_kernel_spec(self.kernel, self.domain)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PosteriorGP:
    gp: GP
    rep: CovRep
    F: ProbeStack
    ctx: OpContext
    solver: LinearSolver
    alpha: Array
    y: Array
    mF: Array

    # Predictive mean at G
    def mean(self, G: ProbeLike) -> Array:
        Gst = as_stack(G)
        mG = self.rep.mean(Gst, self.ctx)                     # (n_G,)
        K_GF = self.rep.cross(Gst, self.F, self.ctx)          # (n_G, n_F)
        return mG + K_GF @ self.alpha

    # Predictive variance diag at G (scalar-output case)
    def variance(self, G: ProbeLike) -> Array:
        Gst = as_stack(G)
        K_GG = self.rep.cross(Gst, Gst, self.ctx)             # (n_G, n_G) dense
        K_GF = self.rep.cross(Gst, self.F, self.ctx)          # (n_G, n_F)
        # Solve (K_FF+Σ)^{-1} K_FG
        K_FG = K_GF.T
        W = self.solver.solve(K_FG)                    # (n_F, n_G)
        # Var = diag(K_GG - K_GF @ W)
        C = K_GF @ W                                          # (n_G, n_G)
        return jnp.clip(jnp.diag(K_GG) - jnp.diag(C), a_min=0.)

    # Full covariance (small n_G)
    def covariance(self, G: ProbeLike) -> Array:
        Gst = as_stack(G)
        K_GG = self.rep.cross(Gst, Gst, self.ctx)             # (n_G, n_G)
        K_GF = self.rep.cross(Gst, self.F, self.ctx)
        K_FG = K_GF.T
        W = self.solver.solve(K_FG)                    # (n_F, n_G)
        return K_GG - K_GF @ W
