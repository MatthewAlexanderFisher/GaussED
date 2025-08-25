from gaussed.gp.backends.registry import register_backend
from gaussed.gp.backends.base import Backend
from gaussed.linops.linops import LinearOperator

@register_backend("features")
def FeaturesBackend(phi) -> Backend:
    class _FeatBackend:
        name = "features"
        def __init__(self): self.phi = phi   # x -> R^M
        def data_operator(self, gp, L):
            A = L.to_features(self.phi)      # (m, M) dense
            # Return operator v -> A (A^T v); keep A around in closure
            return LinearOperator(
                matvec=lambda v: A @ (A.T @ v),
                rmatvec=None, shape=(A.shape[0], A.shape[0])
            )
        def cross_operator_points(self, gp, L, Xstar):
            A = L.to_features(self.phi)      # (m, M)
            Phi_star = self.phi(Xstar)       # (N*, M)
            KstarL = Phi_star @ A.T          # (N*, m)
            return LinearOperator.from_dense(KstarL)
    return _FeatBackend()
