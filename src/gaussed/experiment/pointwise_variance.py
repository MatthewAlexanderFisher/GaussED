import torch

from gaussed.experiment.acquisition import Acquisition
from gaussed.utils.lin_alg_solvers import SafeSVDSolver


class PointwiseVariance(Acquisition):
    def __init__(self, observable, qoi, solver=SafeSVDSolver()):
        self.stochastic = False

        self.observable = observable
        self.qoi = qoi

        self.eval_params = {}

        self.solver = solver

    def eval(self, d, covariance_matrix):
        """Evaluates the posterior variance at given design point and covariance matrix.

        Args:
            d ([torch.Tensor]): [Design point]
            covariance_matrix ([torch.Tensor]): [Posterior covariance matrix]

        Returns:
            [torch.Tensor]: [Posterior pointwise variance]
        """
        M = int(covariance_matrix.shape[0] ** (1 / self.observable.basis.dim))

        s_eig = self.observable.spectral_eig(M)
        basis_eval = (
            self.observable.basis.eval(d, M).mul(s_eig) * self.observable.amplitude
        )

        return -torch.matmul(torch.matmul(basis_eval, covariance_matrix), basis_eval.T)
