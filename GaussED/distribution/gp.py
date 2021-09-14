import torch

from GaussED.distribution.base import Distribution
from GaussED.utils.lin_alg_solvers import DefaultSolver


class GP(Distribution):  # TODO: implement standard GP

    def __init__(self, mean, kernel, solver=DefaultSolver()):
        self.mean = mean
        self.kernel = kernel

        self.solver = solver

        self.dim = self.kernel.dim
        self.t_dim = self.kernel.dim

        self.func_dist = True  # distribution over functions
        self.func_dim = self.kernel.dim

        self.kernel = kernel
        self.dim = self.kernel.dim

    def condition_x(self, x, X, Y, solver=None):

        if solver is None:
            solver = self.solver

        K_XX = self.kernel.eval(X, X)
        K_Xx = self.kernel.eval(X, x)
        K_xx = self.kernel.eval(x, x)

        inverse = solver.inverse(K_XX)

        solved_y = solver.solve(inverse, Y.unsqueeze(1))
        # cholesky_solve(unsqueeze(Y, 1), chol).squeeze(-1)

        solved_gram = solver.solve(inverse, K_Xx)

        mean = torch.matmul(K_Xx.T, solved_y)
        covariance_matrix = K_xx - torch.matmul(K_Xx.T, solved_gram)

        return mean, covariance_matrix

    def get_prior(self, x, differentiate=None):
        if differentiate is True:
            K_xx = self.kernel.deriv_eval(x,x)
        else:
            K_xx = self.kernel.eval(x, x)
        return torch.zeros(x.size(0)), K_xx

    def sample(self, mean, covariance, n):

        m = mean.size(0)

        U, S, V = torch.svd(covariance)
        s_cov = torch.matmul(U, torch.diag(torch.sqrt(S)))

        sn = torch.distributions.MultivariateNormal(torch.zeros(m), torch.eye(m))
        sn_samples = sn.sample(torch.Size([n])).T
        mn_samples = torch.add(torch.matmul(s_cov, sn_samples).T, mean).T

        return mn_samples
