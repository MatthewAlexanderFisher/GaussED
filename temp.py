# # Operator/Functional protocols:

# class Operator(Protocol):
#     def __call__(self, g: FunSpec) -> FunSpec: ...
#     def lift_left(self, ks: KernelSpec, ctx: OpContext) -> KernelSpec: ...
#     def lift_right(self, ks: KernelSpec, ctx: OpContext) -> KernelSpec: ...


# class Functional(Protocol):
#     # unary realisation: FunSpec -> Array
#     def __call__(self, g: FunSpec, ctx: OpContext) -> Array: ...

#     # build a function of Y that returns the left kernel block (n_L, q)
#     def left_kernel_map(self, ks: KernelSpec, ctx: OpContext) -> FunSpec: ...

#     # Consumes a FunSpec (function of Y) and realises it under this functional
#     def right_reduce(self, F: FunSpec, ctx: OpContext) -> Array: ...

#     # default pair implementation (needs to be copied to all classes following this protocol)
#     def pair(self, other: "Functional", ks: KernelSpec, ctx: OpContext) -> Array: ...


# # Probe base class:

# class Probe:
#     ops: Tuple[Operator, ...]
#     fnl: Functional

#     # unary
#     def apply(self, g: FunSpec, ctx: OpContext) -> Array:
#         for op in self.ops: g = op(g)
#         return self.fnl(g, ctx)

#     # binary (kernel)
#     def kernel(self, other: "Probe", ks: KernelSpec, ctx: OpContext) -> Array:
#         # left lifts
#         ksL = ks
#         for op in self.ops:
#             ksL = op.lift_left(ksL, ctx)
#         # right lifts
#         ksLR = ksL
#         for op in other.ops:
#             ksLR = op.lift_right(ksLR, ctx)
#         # realise with the two functionals
#         return self.fnl.pair(other.fnl, ksLR, ctx)


# # GP object

# class GP:
#     domain: Domain
#     codomain: Codomain
#     mean: MeanFun
#     kernel: Kernel
#     backend: Backend
#     op_ctx: Optional[OpContext] = None  # user can set/override

#     def __init__(self, domain: Domain, codomain: Codomain, mean: MeanFun, kernel: Kernel, backend: Backend):
#         self.domain = domain
#         self.codomain = codomain
#         self.mean = mean
#         self.kernel = kernel
#         self.backend = backend

#     # ... need to decide on methods/API


# # GPModel Object

# @dataclass
# class GPModel:
#     gp: GP
#     likelihood: Likelihood = GaussianLikelihood()   # Gaussian, Bernoulli, Poisson, Student-t, ...

#     def condition(self, F: ProbeLike, y: Array, *, backend: "Backend") -> PosteriorGP: ...
#         # Currently only has condition as a place holder
    
#     # need to decide on API and methods (e.g. sample, mean, variance, covariance etc.)


# # Backend

# class Backend(Protocol):
#     name: str
#     solver: LinearSolver

#     def make_specs(self, model: "GPModel") -> Tuple[KernelSpec, FunSpec, OpContext]:
#         ... # Define the specs (FunSpec/KernelSpec/OpContext for linear ops)

#     def make_rep(self, model: "GPModel") -> CovRep:
#         ... # make CovRep (depends on exact implementation)

#     # 

# @dataclass
# class ExactBackend(Backend):
#     name: str = "exact"


#     def make_specs(self, model: "GPModel"):
#         mean_spec: FunSpec   = make_fun_spec(model.gp.mean, model.gp.domain)
#         kernel_spec: KernelSpec = make_kernel_spec(model.gp.kernel, model.gp.domain)
#         ctx = OpContext(domain=model.gp.domain)  # optional
#         return kernel_spec, mean_spec, ctx

#     def make_rep(self, model: "GPModel"):...

# # Cov Representation (full kernel vs. inducing points vs. basis functions)

# class CovRep(Protocol):
#     kernel_spec: KernelSpec
#     mean_spec: FunSpec

#     def gram(self, F: Probe, G: Probe, ctx: OpContext) -> LinearLike:
#         """Return K_{FG} as Array (dense) or LinearOp."""
#         ...

#     def cross(self, F: Probe, G: Probe, ctx: OpContext) -> Array:
#         """Return K_{FG} as dense Array (for prediction)."""
#         ...

#     def mean(self, F: Probe) -> Array:
#         """Return F[mean]."""
#         ...

# @jax.tree_util.register_pytree_node_class
# @dataclass(frozen=True)
# class KernelRep:
#     kernel_spec: KernelSpec
#     mean_spec: FunSpec

#     # exact Gram; return dense; wrap as LinearOp upstream if desired
#     def gram(self, F: Probe, G: Probe, ctx: OpContext) -> LinearLike:
#         K = F.kernel(G, self.kernel_spec, ctx)     # (n_F, n_G) array
#         return K

#     def cross(self, F: Probe, G: Probe, ctx: OpContext) -> Array:
#         return F.kernel(G, self.kernel_spec, ctx)

#     def mean(self, F: Probe, ctx: OpContext) -> Array:
#         return F.apply(self.mean_spec, ctx)[:, 0]


# # LinearSolver

# # Hooks must *read* state and *return* updated state.
# SolveFn  = Callable[[LinearOp, Array, "LinearSolverState"], Tuple[Array, "LinearSolverState"]]
# SqrtFn   = Callable[[LinearOp, Array, "LinearSolverState"], Tuple[Array, "LinearSolverState"]]
# LogdetFn = Callable[[LinearOp, "LinearSolverState"], Tuple[Array, "LinearSolverState"]]

# @jax.tree_util.register_pytree_node_class
# @dataclass(frozen=True)
# class LinearSolverState:
#     cache: Any = None   # arbitrary pytree (e.g., Cholesky L, CG warm start, Lanczos basis)

#     def tree_flatten(self):
#         return (self.cache,), ()
#     @classmethod
#     def tree_unflatten(cls, aux, children):
#         (cache,) = children
#         return cls(cache)

# @jax.tree_util.register_pytree_node_class
# @dataclass(frozen=True)
# class SolverFns:
#     solve: SolveFn
#     sqrt: SqrtFn
#     logdet: LogdetFn

#     def tree_flatten(self):
#         # functions are static aux
#         return (), (self.solve, self.sqrt, self.logdet)
#     @classmethod
#     def tree_unflatten(cls, aux, children):
#         s, q, l = aux
#         return cls(s, q, l)

# # ----- Factor protocol (solver handle) --------------------------------------
# @jax.tree_util.register_pytree_node_class
# @dataclass(frozen=True)
# class LinearSolver:
#     """One solver bound to one operator, with a mutable (functional) state."""
#     op: LinearOp
#     fns: SolverFns
#     state: LinearSolverState = LinearSolverState()

#     # Core API returns (result, updated_solver) so caches propagate.
#     def solve_and_update(self, rhs: Array) -> Tuple[Array, "LinearSolver"]:
#         out, new_state = self.fns.solve(self.op, rhs, self.state)
#         return out, replace(self, state=new_state)

#     def sqrt_and_update(self, rhs: Array) -> Tuple[Array, "LinearSolver"]:
#         out, new_state = self.fns.sqrt(self.op, rhs, self.state)
#         return out, replace(self, state=new_state)

#     def log_det_and_update(self) -> Tuple[Array, "LinearSolver"]:
#         val, new_state = self.fns.logdet(self.op, self.state)
#         return val, replace(self, state=new_state)

#     # Convenience (discard updates) – use sparingly
#     def solve(self, rhs: Array) -> Array:
#         y, _ = self.solve_and_update(rhs)
#         return y
#     def sqrt(self, rhs: Array) -> Array:
#         y, _ = self.sqrt_and_update(rhs)
#         return y
#     def log_det(self) -> Array:
#         y, _ = self.log_det_and_update()
#         return y

#     # Rebind to a new operator (clears cache)
#     def rebind(self, new_op: LinearLike) -> "LinearSolver":
#         return replace(self, op=AsLinearOp(new_op), state=LinearSolverState(None))

#     # expose block-append for Cholesky caches (no-op otherwise)
#     def extend_block(self, B: Array, C: Array, jitter: float = 0.0) -> "LinearSolver":
#         cache = self.state.cache
#         if isinstance(cache, CholCache):
#             new_cache = chol_extend_block(cache, B, C, jitter)
#             return replace(self, state=LinearSolverState(new_cache))
#         # If cache type doesn't support, just return self 
#         return self

