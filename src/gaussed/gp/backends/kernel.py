from gaussed.gp.backends.registry import register_backend
from gaussed.gp.backends.base import Backend
from gaussed.linops.linops import LinearOperator

@register_backend("kernel")
def KernelBackend(**kwargs) -> Backend:
    class _KernelBackend:
        name = "kernel"
        def prepare(self, gp, L_train): return self
        def data_operator(self, gp, L):
            # matrix-free matvec using kernel calls (chunked or KeOps)
            def mv(v): return kernel_matvec(L, gp.kernel, v)   # implement chunked K_LL @ v
            m = L.m
            return LinearOperator(matvec=mv, rmatvec=None, shape=(m, m))
        def cross_operator_points(self, gp, L, Xstar):
            KstarL = L.cross_with_points(Xstar, gp.kernel)     # (N*, m)
            return LinearOperator.from_dense(KstarL)            # or a matvec closure
    return _KernelBackend()
