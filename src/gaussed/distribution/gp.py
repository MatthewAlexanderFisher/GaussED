import torch

from gaussed.distribution.base import Distribution
from gaussed.utils.lin_alg_solvers import DefaultSolver


class GP(Distribution):
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
        """Computes the posterior mean and covariance at locations x, conditioned upon output observations Y, observed at input locations X

        Args:
            x ([torch.Tensor]): [Locations to evaluate posterior mean and covariance amtrix]
            X ([torch.Tensor]): [Tensor of input locations]
            Y ([type]): [Tensor of output observations]
            solver ([Solver], optional): [Solver]. Defaults to None and so to self.solver.

        Returns:
            [torch.Tensor, torch.Tensor]: [Posterior mean and covariance matrix]
        """

        if solver is None:
            solver = self.solver

        K_XX = self.kernel.eval(X, X)
        K_Xx = self.kernel.eval(X, x)
        K_xx = self.kernel.eval(x, x)

        inverse = solver.inverse(K_XX)

        solved_y = solver.solve(inverse, Y.unsqueeze(1))

        solved_gram = solver.solve(inverse, K_Xx)

        mean = torch.matmul(K_Xx.T, solved_y)
        covariance_matrix = K_xx - torch.matmul(K_Xx.T, solved_gram)

        return mean, covariance_matrix

    def get_prior(self, x):
        """Returns prior mean and covariance at locations x.

        Args:
            x ([torch.Tensor]): [Locations to evaluate prior mean and prior covariance]

        Returns:
            [torch.Tensor, torch.Tensor]: [Prior mean and covariance matrix]
        """

        K_xx = self.kernel.eval(x, x)
        prior_mean = self.mean.eval(x)
        return prior_mean, K_xx

    def sample(self, mean, covariance, n):
        """Sample from the GP with given mean and covariance, n number of times.

        Args:
            mean ([torch.Tensor]): [Mean Tensor]
            covariance ([torch.Tensor]): [Covariance matrix Tensor]
            n ([int]): [Number of samples]

        Returns:
            [torch.Tensor]: [GP samples]
        """

        m = mean.size(0)

        U, S, V = torch.svd(covariance)
        s_cov = torch.matmul(U, torch.diag(torch.sqrt(S)))

        sn = torch.distributions.MultivariateNormal(torch.zeros(m), torch.eye(m))
        sn_samples = sn.sample(torch.Size([n])).T
        mn_samples = torch.add(torch.matmul(s_cov, sn_samples).T, mean).T

        return mn_samples
