import torch

from GaussED.experiment.acquisition import Acquisition
from GaussED.utils.lin_alg_solvers import SafeSVDSolver


class PointwiseVariance(Acquisition):

    def __init__(self, observable, qoi, solver=SafeSVDSolver()):
        self.stochastic = False

        self.observable = observable
        self.qoi = qoi

        self.eval_params = {}

        self.solver = solver

    def eval(self, x, covariance_matrix):
        M = int(covariance_matrix.shape[0] ** (1 / self.observable.basis.dim))

        s_eig = self.observable.spectral_eig(M)
        basis_eval = self.observable.basis.eval(x, M).mul(s_eig) * self.observable.amplitude

        return -torch.matmul(torch.matmul(basis_eval, covariance_matrix), basis_eval.T)
