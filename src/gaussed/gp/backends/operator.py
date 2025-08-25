from gaussed.gp.backends.registry import register_backend
from gaussed.gp.backends.base import Backend
from gaussed.linops.linops import LinearOperator

@register_backend("operator")
def OperatorBackend(basis, C_op) -> Backend:
    class _OpBackend:
        name = "operator"
        def __init__(self): self.basis = basis; self.C = C_op
        def prepare(self, gp, L_train): return self
        def data_operator(self, gp, L):
            A = L.to_basis(self.basis)               # (m x n) LinearOperator
            return sandwich(A, self.C)               # A ∘ C ∘ A^T
        def cross_operator_points(self, gp, L, Xstar):
            A = L.to_basis(self.basis)
            Astar = points_to_basis(self.basis, Xstar)          # (N* x n) LinearOperator
            return compose(Astar, compose(self.C, A.T))         # maps R^m -> R^{N*}
    return _OpBackend()