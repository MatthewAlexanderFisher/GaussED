# Base Backend


@dataclass # a backend should be supplied to the GP model by default (perhaps having a probe context is neater... but could create a constructor for it)
class Backend(Protocol):
    # don't want to define these on an init (should be actually supplied concretely for other classes)
    name: str = "base_backend"
    solver: Solver # the Solver (supplies the numerical algorithms e.g. integration and the linear solvers)
    covrep: CovRep # the "type" of the GP (full kernel/inducing/basis)
    
    ctx: ProbeContext

    # manual context: all additional context can be stored here e.g.
    # dx: Optional[Callable[[Array, Array, int], Array]] = None             # ∂/∂x_a k(X,Y)
    # dy: Optional[Callable[[Array, Array, int], Array]] = None             # ∂/∂y_a k(X,Y)
    # dxy: Optional[Callable[[Array, Array, int, int], Array]] = None       # ∂²/∂x_a∂y_b k(X,Y)
    # phi: Optional["BasisMap"] = None                                     # expose Φ and its derivatives

    def condition(self, model: GPModel, F: Probe, y: Array) -> PosteriorGP: ... # performs the numerical conditioning

    def construct_ctx(self, model: GPModel) -> ProbeContext: ...
        # e.g. (will depend on type of Backend)
        # if GPModel.kernel has dx: ...
        

@dataclass
class BasisBackend:
    name: str = "basis_backend"
    solver: Solver
    covrep: CovRep = BasisRep()

    def __init__(self, basis: BasisMap):
        self.basis = basis

    def construct_ctx(self, model: GPModel) -> ProbeContext: ...
        # e.g. (will depend on type of Backend)
        # if self.basis has dx: dx = self.basis.dx
        # ctx = ProbeContext(basis_dx = self.basis.dx)



# The probes only implement one apply. E.g.
@dataclass
class GradProbe:
    X: Array
    axis: int
    def n(self): return self.X.shape[0]
    def mean(self, mean_fn): return mean_fn.grad(self.X, self.axis)

    def apply(self, func, ctx):
        def g(Y):
            J = jax.jacfwd(func)(self.X)                       # (n, d, |Y|)
            return J[:, self.axis, :]
        return g


# I guess the CovRep could define what an apply_left or apply_right does:

class KernelRep(CovRep):
    kernel: Kernel

    def apply_left(self, p: Probe, ctx):
        func = lambda Y: self.kernel(X, Y)

        return p.apply(func, ctx)

    def apply_right(self, p: Probe, ctx):
        func = lambda X: self.kernel(X, Y)

        return p.apply(func, ctx)

@dataclass
class BasisRep(CovRep):
    basis: BasisMap         # phi.design(Probe) -> Φ (n×m)

    # since a basis based GP should just compute Φ (n×m) - its derivatives are better?

    def apply_left(self, p: Probe, ctx) -> Callable:

        return p.apply(lambda Y: self.basis.design(Y), ctx)
    
    def apply_right(self, p: Probe, ctx) -> Callable:

        return p.apply(lambda Y: self.basis.design(Y), ctx)

    # the implementation of a basis backend should handle how to combine these