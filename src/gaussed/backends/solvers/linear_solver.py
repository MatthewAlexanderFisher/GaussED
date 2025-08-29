from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Protocol, Optional, Callable, Any, Tuple
import jax.numpy as jnp
from jax import Array
import jax

from gaussed.linops.linop import LinearOp, DenseOp, AsLinearOp
from gaussed.types import LinearLike

# -------------------------------------------------------------------
# Stateful solver: mix-and-match hooks + cache
# -------------------------------------------------------------------

# Hooks must *read* state and *return* updated state.
SolveFn  = Callable[[LinearOp, LinearOp, "LinearSolverState"], Tuple[LinearOp, "LinearSolverState"]]
SqrtFn   = Callable[[LinearOp, LinearOp, "LinearSolverState"], Tuple[LinearOp, "LinearSolverState"]]
LogdetFn = Callable[[LinearOp, "LinearSolverState"], Tuple[Array, "LinearSolverState"]]

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class LinearSolverState:
    cache: Any = None   # arbitrary pytree (e.g., Cholesky L, CG warm start, Lanczos basis)

    def tree_flatten(self):
        return (self.cache,), ()
    @classmethod
    def tree_unflatten(cls, aux, children):
        (cache,) = children
        return cls(cache)

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SolverFns:
    solve: SolveFn
    sqrt: SqrtFn
    logdet: LogdetFn

    def tree_flatten(self):
        # functions are static aux
        return (), (self.solve, self.sqrt, self.logdet)
    @classmethod
    def tree_unflatten(cls, aux, children):
        s, q, l = aux
        return cls(s, q, l)

# ----- Factor protocol (solver handle) --------------------------------------
@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class LinearSolver:
    """One solver bound to one operator, with a mutable (functional) state."""
    op: LinearOp
    fns: SolverFns
    state: LinearSolverState = LinearSolverState()

    # Core API returns (result, updated_solver) so caches propagate.
    def solve_and_update(self, rhs: LinearOp) -> Tuple[LinearOp, "LinearSolver"]:
        out, new_state = self.fns.solve(self.op, rhs, self.state)
        return out, replace(self, state=new_state)

    def sqrt_and_update(self, rhs: LinearOp) -> Tuple[LinearOp, "LinearSolver"]:
        out, new_state = self.fns.sqrt(self.op, rhs, self.state)
        return out, replace(self, state=new_state)

    def log_det_and_update(self) -> Tuple[Array, "LinearSolver"]:
        val, new_state = self.fns.logdet(self.op, self.state)
        return val, replace(self, state=new_state)

    # Convenience (discard updates) – use sparingly
    def solve(self, rhs: LinearOp) -> LinearOp:
        y, _ = self.solve_and_update(rhs)
        return y
    def sqrt(self, rhs: LinearOp) -> LinearOp:
        y, _ = self.sqrt_and_update(rhs)
        return y
    def log_det(self) -> Array:
        y, _ = self.log_det_and_update()
        return y

    # Rebind to a new operator (clears cache)
    def rebind(self, new_op: LinearLike) -> "LinearSolver":
        return replace(self, op=AsLinearOp(new_op), state=LinearSolverState(None))

    # expose block-append for Cholesky caches (no-op otherwise)
    # def extend_block(self, B: Array, C: Array, jitter: float = 0.0) -> "LinearSolver":
    #     cache = self.state.cache
    #     if isinstance(cache, CholCache):
    #         new_cache = chol_extend_block(cache, B, C, jitter)
    #         return replace(self, state=LinearSolverState(new_cache))
    #     # If cache type doesn't support, just return self 
    #     return self

    # pytree
    def tree_flatten(self):
        children = (self.op, self.state)
        aux = (self.fns,)
        return children, aux
    @classmethod
    def tree_unflatten(cls, aux, children):
        (fns,) = aux
        op, st = children
        return cls(op, fns, st)


# Solver factory
def make_solver(
    A: LinearLike,
    *,
    solve: SolveFn,
    sqrt: SqrtFn,
    logdet: LogdetFn,
    init_cache: Any = None,
) -> LinearSolver:
    op = AsLinearOp(A)
    fns = SolverFns(solve=solve, sqrt=sqrt, logdet=logdet)
    return LinearSolver(op=op, fns=fns, state=LinearSolverState(init_cache))
