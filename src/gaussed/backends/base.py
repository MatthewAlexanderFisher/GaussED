from __future__ import annotations
from typing import Protocol, Tuple, Optional, TYPE_CHECKING
from dataclasses import dataclass
from jax import Array
import jax

if TYPE_CHECKING:
    from gaussed.model import GPModel
    from gaussed.backends.solvers.quadrature import Quadrature, BiQuadrature
    from gaussed.backends.reps.base import CovRep

from gaussed.utils.make_specs import make_kernel_spec, make_fun_spec
from gaussed.gp.gp_ops.base import KernelSpec, FunSpec, OpContext
from gaussed.backends.reps.kernel_rep import KernelRep
from gaussed.backends.solvers.linear_solver import SolverFns, LinearSolverState


# Base Backend
class Backend(Protocol):
    solverfns: SolverFns
    solver_state: LinearSolverState = LinearSolverState()

    def make_specs(self, model: "GPModel") -> Tuple[KernelSpec, FunSpec, OpContext]:
        ... # Define the specs (FunSpec/KernelSpec/OpContext for linear ops)

    def make_rep(self, model: "GPModel") -> CovRep:
        ... # make CovRep (depends on exact implementation)

    # 

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ExactBackend(Backend):
    solverfns: SolverFns
    solver_state: LinearSolverState = LinearSolverState()

    quad: Optional["Quadrature"] = None
    quad_xy: Optional["BiQuadrature"] = None  # for double integrals
    quad_x: Optional["Quadrature"] = None  # for single integrals
    quad_y: Optional["Quadrature"] = None  # for single integrals


    def make_specs(self, model: "GPModel"):
        mean_spec   = make_fun_spec(model.gp.mean, model.gp.domain)
        kernel_spec = make_kernel_spec(model.gp.kernel, model.gp.domain)
        ctx = OpContext(domain=model.gp.domain, quad=self.quad, quad_xy=self.quad_xy, quad_x=self.quad_x, quad_y=self.quad_y)  # optional
        return kernel_spec, mean_spec, ctx

    def make_rep(self, model: "GPModel") -> CovRep:
        ks, ms, ctx = self.make_specs(model)
        return KernelRep(ks, ms, ctx)


    # ---- PyTree plumbing -----------------------------------------------------
    def tree_flatten(self):
        """
        Children: none (all fields are static config).
        Aux: everything (strings/objects/callables).
        """
        children = ()
        aux = (self.solverfns, self.solver_state, self.quad, self.quad_xy, self.quad_x, self.quad_y)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        solverfns, solver_state, quad, quad_xy, quad_x, quad_y = aux
        return cls(solverfns=solverfns, solver_state=solver_state, quad=quad, quad_xy=quad_xy, quad_x=quad_x, quad_y=quad_y)

    @classmethod
    def axes(cls):
        """
        Create an 'axes' instance for shape/dtype metadata.
        Since the dataclass is frozen, use object.__setattr__.
        """
        obj = object.__new__(cls)
        object.__setattr__(obj, "solver", None)
        object.__setattr__(obj, "quad", None)
        object.__setattr__(obj, "quad_xy", None)
        object.__setattr__(obj, "quad_x", None)
        object.__setattr__(obj, "quad_y", None)
        return obj
