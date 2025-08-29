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


import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Optional, Tuple

# --- GP ----------------------------------------------------------------------

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GP:
    domain: "Domain"
    codomain: "Codomain"
    mean: "MeanFun"        # pytree (params are leaves)
    kernel: "Kernel"       # pytree (params are leaves)
    backend: "Backend"     # pytree (solver state may be a leaf)
    op_ctx: Optional["OpContext"] = None  # usually static config

    # Bridge to specs
    def mean_spec(self) -> "FunSpec":
        return make_fun_spec(self.mean, self.domain)

    def kernel_spec(self) -> "KernelSpec":
        return make_kernel_spec(self.kernel, self.domain)

    # --- pytree plumbing ---
    # Children: mean, kernel, backend (to trace their params/state)
    # Aux: domain, codomain, op_ctx (static configs)
    def tree_flatten(self):
        children = (self.mean, self.kernel, self.backend)
        aux = (self.domain, self.codomain, self.op_ctx)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        domain, codomain, op_ctx = aux
        mean, kernel, backend = children
        # frozen=True -> construct then set fields via __setattr__
        obj = object.__new__(cls)
        object.__setattr__(obj, "domain", domain)
        object.__setattr__(obj, "codomain", codomain)
        object.__setattr__(obj, "mean", mean)
        object.__setattr__(obj, "kernel", kernel)
        object.__setattr__(obj, "backend", backend)
        object.__setattr__(obj, "op_ctx", op_ctx)
        return obj

    @classmethod
    def axes(cls, mean_axes: "MeanFun", kernel_axes: "Kernel", backend_axes: "Backend"):
        """Axis-spec helper if you use custom-named axes for params/state."""
        obj = object.__new__(cls)
        object.__setattr__(obj, "domain", None)
        object.__setattr__(obj, "codomain", None)
        object.__setattr__(obj, "mean", mean_axes)
        object.__setattr__(obj, "kernel", kernel_axes)
        object.__setattr__(obj, "backend", backend_axes)
        object.__setattr__(obj, "op_ctx", None)
        return obj


# --- PosteriorGP -------------------------------------------------------------

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PosteriorGP:
    gp: GP                 # child (contains params + backend state)
    rep: "CovRep"          # child (holds specs; usually params are children)
    F: "ProbeStack"        # child
    ctx: "OpContext"       # aux (static config)
    solver: "LinearSolver" # child (stateful cache)
    alpha: jnp.ndarray     # child
    y: jnp.ndarray         # child
    mF: jnp.ndarray        # child

    # Predictive mean at G
    def mean(self, G: "ProbeLike") -> jnp.ndarray:
        Gst = as_stack(G)
        mG = self.rep.mean(Gst, self.ctx)                      # (n_G,)
        K_GF = self.rep.cross(Gst, self.F, self.ctx)           # (n_G, n_F)
        return mG + K_GF @ self.alpha

    # Predictive variance diag at G (scalar-output case)
    def variance(self, G: "ProbeLike") -> jnp.ndarray:
        Gst = as_stack(G)
        K_GG = self.rep.cross(Gst, Gst, self.ctx)              # (n_G, n_G)
        K_GF = self.rep.cross(Gst, self.F, self.ctx)           # (n_G, n_F)
        # Solve (K_FF+Σ)^{-1} K_FG, with multiple RHS columns
        K_FG = K_GF.T                                          # (n_F, n_G)
        W = self.solver.solve(K_FG)                     # (n_F, n_G)
        C = K_GF @ W                                           # (n_G, n_G)
        return jnp.clip(jnp.diag(K_GG) - jnp.diag(C), a_min=0.)

    # Full covariance (small n_G)
    def covariance(self, G: "ProbeLike") -> jnp.ndarray:
        Gst = as_stack(G)
        K_GG = self.rep.cross(Gst, Gst, self.ctx)              # (n_G, n_G)
        K_GF = self.rep.cross(Gst, self.F, self.ctx)           # (n_G, n_F)
        K_FG = K_GF.T
        W = self.solver.solve(K_FG)                     # (n_F, n_G)
        return K_GG - K_GF @ W

    # --- pytree plumbing ---
    def tree_flatten(self):
        children = (self.gp, self.rep, self.F, self.solver, self.alpha, self.y, self.mF)
        aux = (self.ctx,)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (ctx,) = aux
        gp, rep, F, solver, alpha, y, mF = children
        obj = object.__new__(cls)
        object.__setattr__(obj, "gp", gp)
        object.__setattr__(obj, "rep", rep)
        object.__setattr__(obj, "F", F)
        object.__setattr__(obj, "ctx", ctx)
        object.__setattr__(obj, "solver", solver)
        object.__setattr__(obj, "alpha", alpha)
        object.__setattr__(obj, "y", y)
        object.__setattr__(obj, "mF", mF)
        return obj

    @classmethod
    def axes(cls, gp_axes: GP, rep_axes: "CovRep", F_axes: "ProbeStack", solver_axes: "LinearSolver"):
        obj = object.__new__(cls)
        object.__setattr__(obj, "gp", gp_axes)
        object.__setattr__(obj, "rep", rep_axes)
        object.__setattr__(obj, "F", F_axes)
        object.__setattr__(obj, "ctx", None)
        object.__setattr__(obj, "solver", solver_axes)
        object.__setattr__(obj, "alpha", None)
        object.__setattr__(obj, "y", None)
        object.__setattr__(obj, "mF", None)
        return obj
