from __future__ import annotations
from typing import Protocol, Tuple, Optional, TYPE_CHECKING
from dataclasses import dataclass, field, replace
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
from gaussed.linops.constructors import LinOpConstructor, DenseGramConstructor


# Base Backend
class Backend(Protocol):
    solverfns: SolverFns
    solver_state: LinearSolverState = LinearSolverState()

    # the linop constructor:
    linop_constructor: LinOpConstructor

    def make_specs(self, model: "GPModel") -> Tuple[KernelSpec, FunSpec, OpContext]:
        ... # Define the specs (FunSpec/KernelSpec/OpContext for linear ops)

    def make_rep(self, model: "GPModel") -> CovRep:
        ... # make CovRep (depends on exact implementation)



@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ExactBackend(Backend):
    solverfns: SolverFns
    solver_state: "LinearSolverState" = field(default_factory=lambda: LinearSolverState())
    linop_constructor: LinOpConstructor = field(default_factory=lambda: DenseGramConstructor())

    quad: Optional["Quadrature"] = None
    quad_xy: Optional["BiQuadrature"] = None  # for double integrals
    quad_x: Optional["Quadrature"] = None  # for single integrals
    quad_y: Optional["Quadrature"] = None  # for single integrals


    def make_specs(self, model: "GPModel"):
        mean_spec   = make_fun_spec(model.gp.mean, model.gp.domain, model.gp.codomain)
        kernel_spec = make_kernel_spec(model.gp.kernel, model.gp.domain, model.gp.codomain)
        ctx = OpContext(domain=model.gp.domain, codomain=model.gp.codomain, quad=self.quad, quad_xy=self.quad_xy, quad_x=self.quad_x, quad_y=self.quad_y)  # optional
        return kernel_spec, mean_spec, ctx

    def make_rep(self, model: "GPModel") -> CovRep:
        ks, ms, ctx = self.make_specs(model)
        return KernelRep(_kernel_spec=ks, _mean_spec=ms, ctx=ctx, linop_constructor=self.linop_constructor)

    # immutable “setter” to swap constructors
    def with_linop_constructor(self, ctor: LinOpConstructor) -> "ExactBackend":
        return replace(self, linop_constructor=ctor)


    # ---- PyTree plumbing -------------------------------------------
    def tree_flatten(self):
        children = (self.solver_state,)
        aux = (self.solverfns, self.linop_constructor, self.quad, self.quad_xy, self.quad_x, self.quad_y)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        solverfns, linop_ctor, quad, quad_xy, quad_x, quad_y = aux
        (solver_state,) = children
        return cls(solverfns=solverfns,
                   solver_state=solver_state,
                   linop_constructor=linop_ctor,
                   quad=quad, quad_xy=quad_xy, quad_x=quad_x, quad_y=quad_y)
