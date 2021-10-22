import torch
from torch.distributions import Normal

from gaussed.experiment.acquisition import Acquisition
from gaussed.utils.lin_alg_solvers import SafeCholeskySolver


class ExpectedImprovement(Acquisition):
    def __init__(
        self, gp, design=None, solver=SafeCholeskySolver(), nugget=None, alpha=None
    ):
        super().__init__(design)

        self.gp = gp
        self.solver = solver
        self.nugget = nugget

        self.mean = None
        self.covariance_matrix = None
        self.y = None  # this is simply to check if there has been update

        self.normal = Normal(torch.Tensor([0]), torch.Tensor([1]))
        if alpha is None:
            self.alpha = 0.01
        else:
            self.alpha = alpha

        self.eval_params = {"m": 10}

    def eval(self, D, phi_mat, y, m):
        """Evaluate expected improvement using its closed form.

        Args:
            d ([torch.Tensor]): [Design point]
            phi_mat ([torch.Tensor]): [Matrix of basis function evaluations]
            y ([torch.Tensor]): [Output observations]
            m ([list, int]): [Number of basis functions]

        Returns:
            [torch.Tensor]: [Expected Improvement]
        """

        current_max = torch.max(y)

        if self.design is not None:
            d = self.design.transform(D)
        else:
            d = D

        if d.dim() == 1:
            d_t = d.unsqueeze(0)
        else:
            d_t = d

        if self.mean is None and self.covariance_matrix is None:
            self.update_cache(phi_mat, y, m)
        else:
            if not torch.equal(
                y, self.y
            ):  # if we have new data point update cached mean and covariance matrix
                self.update_cache(phi_mat, y, m)

        mean, covariance_matrix = self.mean, self.covariance_matrix

        mean_func = self.gp.get_mean(mean, m=m)
        var_func = self.gp.get_variance(covariance_matrix, m=m)

        Z = (mean_func(d_t) - current_max - self.alpha) / var_func(d_t).sqrt()

        ei = (mean_func(d_t) - current_max - self.alpha) * self.normal.cdf(
            Z
        ) + var_func(d_t).sqrt() * torch.exp(self.normal.log_prob(Z))

        return -ei

    def set_eval_params(self, eval_params):
        """Set the default parameters of the eval function.

        Args:
            eval_params ([dict]): [Dictionary of parameters to be passed to eval]
        """

        self.eval_params = eval_params

    def update_cache(self, phi_mat, y, m):
        """Update cached phi_mat, y and m

        Args:
            phi_mat ([torch.Tensor]): [Matrix of basis function evaluations]
            y ([torch.Tensor]): [Output observations]
            m ([int, list]): [Number of basis functions]
        """
        mean, covariance_matrix = self.gp.condition(phi_mat, y, solver=self.solver, m=m)
        self.mean, self.covariance_matrix = mean, covariance_matrix
        self.y = y


class StochasticExpectedImprovement(Acquisition):
    def __init__(self, gp, design=None, solver=SafeCholeskySolver(), nugget=None):
        super().__init__(design)

        self.gp = gp
        self.solver = solver
        self.nugget = nugget

        self.eval_params = {"m": 10, "n": 30}

    def eval(self, D, phi_mat, y, m, n=50):
        """Evaluate expected improvement using a stochastic approximation.

        Args:
            d ([torch.Tensor]): [Design point]
            phi_mat ([torch.Tensor]): [Matrix of basis function evaluations]
            y ([torch.Tensor]): [Output observations]
            m ([list, int]): [Number of basis functions to use]
            n (int, optional): [Batch size of stochastic approximation]. Defaults to 50.

        Returns:
            [torch.Tensor]: [Stochastic Expected Improvement]
        """

        if self.design is not None:
            d = self.design.transform(D)
        else:
            d = D

        current_max = torch.max(y)

        mean, cov = self.gp.condition(phi_mat, y, solver=self.solver, m=m)

        samples_d = torch.clamp(
            self.gp.sample_mesh(mean, cov, d.unsqueeze(1), n, solver=self.solver)
            - current_max,
            0,
        )

        return -torch.mean(samples_d, dim=1)

    def set_eval_params(self, eval_params):
        """Set the default parameters of the eval function.

        Args:
            eval_params ([dict]): [Dictionary of parameters to be passed to eval]
        """
        self.eval_params = eval_params
