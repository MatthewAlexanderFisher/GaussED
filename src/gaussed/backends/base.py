from __future__ import annotations
from typing import Protocol, Tuple, Optional
from dataclasses import dataclass
from jax import Array
import jax


from gaussed.types import LinearLike
from gaussed.gp.gp_ops.base import KernelSpec, FunSpec, OpContext
from gaussed.model import GPModel
from gaussed.backends.reps.base import CovRep
from gaussed.backends.reps.kernel_rep import KernelRep
from gaussed.backends.solvers.linear_solver import LinearSolver
from gaussed.utils.make_specs import make_kernel_spec, make_fun_spec
from gaussed.backends.solvers.quadrature import Quadrature, BiQuadrature


# Base Backend
class Backend(Protocol):
    name: str
    solver: LinearSolver

    def make_specs(self, model: "GPModel") -> Tuple[KernelSpec, FunSpec, OpContext]:
        ... # Define the specs (FunSpec/KernelSpec/OpContext for linear ops)

    def make_rep(self, model: "GPModel") -> CovRep:
        ... # make CovRep (depends on exact implementation)

    # 

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ExactBackend(Backend):
    name: str = "exact"

    quad: Optional[Quadrature] = None
    quad_xy: Optional[BiQuadrature] = None  # for double integrals
    quad_x: Optional[Quadrature] = None  # for single integrals
    quad_y: Optional[Quadrature] = None  # for single integrals


    def make_specs(self, model: "GPModel"):
        mean_spec   = make_fun_spec(model.gp.mean, model.gp.domain)
        kernel_spec = make_kernel_spec(model.gp.kernel, model.gp.domain)
        ctx = OpContext(domain=model.gp.domain, quad=self.quad, quad_xy=self.quad_xy, quad_x=self.quad_x, quad_y=self.quad_y)  # optional
        return kernel_spec, mean_spec, ctx

    def make_rep(self, model: "GPModel") -> CovRep:
        ks, ms, ctx = self.make_specs(model)
        return KernelRep(ks, ms, ctx)
