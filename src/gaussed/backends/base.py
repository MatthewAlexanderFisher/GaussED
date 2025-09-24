from __future__ import annotations
from typing import Protocol, Tuple, Optional, TYPE_CHECKING, Callable, Any, Union
from dataclasses import dataclass, field, replace
from jax import Array
import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from gaussed.model import GPModel
    from gaussed.backends.solvers.quadrature import Quadrature, BiQuadrature
    from gaussed.backends.reps.base import CovRep

from gaussed.utils.make_specs import make_kernel_spec, make_fun_spec, make_mean_spec
from gaussed.gp.gp_ops.base import KernelSpec, FunSpec, OpContext, BasisSpec
from gaussed.backends.reps.kernel_rep import KernelRep
from gaussed.backends.reps.basis_rep import BasisRep
from gaussed.backends.solvers.linear_solver import SolverFns, LinearSolverState
from gaussed.linops.constructors import LinOpConstructor, DenseGramConstructor, ColaConstructor


# Base Backend
class Backend(Protocol):
    solverfns: SolverFns
    solver_state: LinearSolverState = LinearSolverState()

    # the linop constructor:
    linop_constructor: LinOpConstructor

    def make_specs(self, model: "GPModel") -> Tuple[Union[KernelSpec, FunSpec, BasisSpec], FunSpec, OpContext]:
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
        mean_spec   = make_mean_spec(model.gp.mean, model.gp.domain, model.gp.codomain)
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



@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class BasisBackend(Backend):
    """
    Backend that constructs a BasisRep using a feature map φ and diagonal/full Λ.

    Parameters
    ----------
    basis_family : callable
        Something like LaplaceBasis(dim) that can be called as basis(X, m_per_dim=..., [L=...]) -> (n, m).
    m_per_dim : int
        Number of basis functions per dimension (1D: total m = m_per_dim).
    lambda_builder : Optional[Callable]
        Function (kernel, L, m_per_dim, domain) -> Array for Λ (scalar, (m,), or (m,m)).
        If None, we try: basis.lambd(...) if present, else a default Laplace 1D construction.
    linop_constructor : LinOpConstructor
        Defaults to ColaConstructor() so ΦΛΦᵀ is applied efficiently (optionally CoLA-attached).
    """
    solverfns: SolverFns
    basis_family: Any
    m_per_dim: int
    lambda_builder: Optional[Callable[[Any, int, Any], Array]] = None

    solver_state: "LinearSolverState" = field(default_factory=lambda: LinearSolverState())
    linop_constructor: LinOpConstructor = field(default_factory=DenseGramConstructor)

    quad: Optional["Quadrature"] = None
    quad_xy: Optional["BiQuadrature"] = None
    quad_x: Optional["Quadrature"] = None
    quad_y: Optional["Quadrature"] = None

    # ---- spec building -----------------------------------------------------

    def make_specs(self, model: "GPModel"):
        domain   = model.gp.domain
        codomain = model.gp.codomain

        # mean
        mean_spec = make_mean_spec(model.gp.mean, domain, codomain)

        # φ FunSpec (your wrapper around basis_family(X, m_per_dim))
        phi_fn     = _wrap_basis_call(self.basis_family, self.m_per_dim)
        phi_spec   = make_fun_spec(phi_fn, domain, codomain)  # φ: (n,m)

        # Λ from the same basis + kernel
        Lam = self._make_lambda(model, domain)  # scalar | (m,) | (m,m)

        # >>> Hand back a BasisSpec, not a FunSpec
        ks_or_phi = BasisSpec(phi_spec=phi_spec, Lambda=Lam)

        ctx = OpContext(domain=domain, codomain=codomain,
                        quad=self.quad, quad_xy=self.quad_xy, quad_x=self.quad_x, quad_y=self.quad_y)
        return ks_or_phi, mean_spec, ctx

    def _make_lambda(self, model: "GPModel", domain: Any) -> Array:
        # Caller-provided lambda_builder takes precedence
        if self.lambda_builder is not None:
            return self.lambda_builder(model.gp.kernel, self.m_per_dim, domain)

        return self.basis_family.lambd(self.m_per_dim, model.gp.kernel)  # Laplace-style


    def make_rep(self, model: "GPModel") -> "BasisRep":
        ks_or_phi, ms, ctx = self.make_specs(model)

        # Decide L for Λ (same logic as φ)
        Lam = self._make_lambda(model, model.gp.domain)  # scalar|(m,)|(m,m)

        # Build a real KernelSpec solely to satisfy CovRep.kernel_spec
        ks_parity = make_kernel_spec(model.gp.kernel, model.gp.domain, model.gp.codomain)

        # Plug into BasisRep and delegate matvecs to the chosen constructor
        return BasisRep(kernel_spec=ks_parity,
                        mean_spec=ms,
                        phi_spec=ks_or_phi,     # FunSpec or wrap as BasisSpec if you prefer
                        Lambda=Lam,
                        linop_constructor=self.linop_constructor)

    # immutable “setter” to swap constructors
    def with_linop_constructor(self, ctor: LinOpConstructor) -> "BasisBackend":
        return replace(self, linop_constructor=ctor)

    # ---- PyTree plumbing ---------------------------------------------------
    def tree_flatten(self):
        # arrays or large dynamic state as children
        children = (self.solver_state,)
        aux = (self.solverfns, self.basis_family, self.m_per_dim,
               self.lambda_builder, self.linop_constructor,
               self.quad, self.quad_xy, self.quad_x, self.quad_y)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (solver_state,) = children
        (solverfns, basis_family, m_per_dim, L,
         lambda_builder, linop_ctor, quad, quad_xy, quad_x, quad_y) = aux
        return cls(solverfns=solverfns,
                   basis_family=basis_family,
                   m_per_dim=m_per_dim,
                   lambda_builder=lambda_builder,
                   solver_state=solver_state,
                   linop_constructor=linop_ctor,
                   quad=quad, quad_xy=quad_xy, quad_x=quad_x, quad_y=quad_y)
    


def _wrap_basis_call(basis: Any, m_per_dim: int) -> Callable[[Array], Array]:
    """
    Return φ(X) callable. 
    This runs at build-time (not traced), so the try/except is fine.
    """
    def phi_X(X: Array) -> Array:
        return basis(X, m_per_dim=m_per_dim)

    return phi_X
