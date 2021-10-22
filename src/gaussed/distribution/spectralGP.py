import torch
from torch import unsqueeze
from torch.distributions import MultivariateNormal

import math

import gaussed.transforms.linear as linear
from gaussed.distribution.gp import GP
from gaussed.mean.base import ZeroMean
from gaussed.utils.lin_alg_solvers import DefaultSolver
from gaussed.utils.summation_tensor_gen import sum_tensor_gen


class SpectralGP(GP):
    # keep a copy of the ids of all class instances using a class variable, each list in this list will be a list of
    # all GPs that are transformations of each other.
    t_ids = []

    def __init__(
        self, kernel, mean=None, solver=DefaultSolver(), transform=False, add_id=False
    ):

        super().__init__(None, kernel)

        if add_id is True:
            self._add_id(transform)  # if ID not in the set of ids add a new list of ids

        if mean is None:
            self.mean = ZeroMean(self.dim)

        self.domain = torch.Tensor([[0, 1]] * self.dim)

        self.solver = solver

    def __add__(self, other):
        return linear.Sum(self, other)()

    def _add_id(self, transform):
        for i in SpectralGP.t_ids:
            for j in i:
                if id(self) is j:
                    return
        if transform is False:
            SpectralGP.t_ids.append([id(self)])

    def log_likelihood(
        self, phi_mat, y, solver=None, nugget=None, parameters=None, m=None
    ):
        """Computes the log-likelihood of given phi_mat and observations y.

        Args:
            phi_mat ([type]): [description]
            y ([type]): [description]
            solver ([type], optional): [description]. Defaults to None.
            nugget ([torch.Tensor], optional): [Nugget term that adds to diagonal of the covariance matrix]. Defaults to None.
            parameters ([type], optional): [description]. Defaults to None.
            m ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """

        n, M = phi_mat.shape

        if m is None:
            m = int(M ** (1 / self.kernel.dim))

        if solver is None:
            solver = self.solver

        if parameters is None:
            # for some reason, passing kernel parameters to spectral_eig results in an unexpected gradient error
            # when using a composite kernel like an ANOVA kernel - only the parameters in the first constituent kernel
            # obtain a gradient. not using parameters is a short term fix.
            parameters = self.kernel.parameters

        s_eig = self.kernel.spectral_eig(m)
        Kyy = torch.matmul(phi_mat.mul(s_eig), phi_mat.T)
        Kyy = (Kyy + Kyy.T) / 2

        if nugget is not None:
            Kyy = Kyy + torch.eye(n) * nugget
        else:
            Kyy = Kyy

        if n > M and nugget is not None:  # Woodbury matrix identity
            term = solver.inverse(
                torch.diag(s_eig.pow(-1))
                + 1 / nugget * torch.matmul(phi_mat.T, phi_mat)
            )
            solved = solver.solve(term, phi_mat.T)
            inverse = 1 / nugget * torch.eye(phi_mat.shape[0]) - 1 / (
                nugget ** 2
            ) * torch.matmul(phi_mat, solved)
            solved_y = torch.matmul(inverse, y)
        else:
            inverse = solver.inverse(
                Kyy
            )  # this is usually the Cholesky decomp - not the actual inverse
            solved_y = solver.solve(inverse, y.unsqueeze(1))

        log_det_cov = solver.log_det(Kyy, inverse)
        exponent = torch.dot(y, solved_y)

        return (
            -1
            / 2
            * (n * torch.log(torch.Tensor([2 * math.pi])) + log_det_cov + exponent)
        )

    def log_likelihood_x(self, x, y, m, solver=None, nugget=None, parameters=None):
        """Computes the log-likelihood of observations y at input locations x

        Args:
            x ([torch.Tensor]): [description]
            y ([type]): [description]
            m ([type]): [description]
            solver ([type], optional): [description]. Defaults to None.
            nugget ([torch.Tensor], optional): [Nugget term that adds to diagonal of the covariance matrix]. Defaults to None.
            parameters ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        phi_mat = self.basis_matrix(x, m)
        return self.log_likelihood(
            phi_mat, y, solver=solver, nugget=nugget, parameters=parameters, m=m
        )

    def condition(self, phi_mat, y, solver=None, nugget=None, debug=False, m=None):

        if solver is None:
            solver = self.solver

        n, M = phi_mat.shape
        if m is None:
            m = int(M ** (1 / self.kernel.dim))

        s_eig = self.kernel.spectral_eig(m)

        if n > M and nugget is not None:  # Woodbury matrix identity
            term = solver.inverse(
                torch.diag(s_eig.pow(-1))
                + 1 / nugget * torch.matmul(phi_mat.T, phi_mat)
            )
            solved = solver.solve(term, phi_mat.T)
            inverse = 1 / nugget * torch.eye(phi_mat.shape[0]) - 1 / (
                nugget ** 2
            ) * torch.matmul(phi_mat, solved)

            if debug:
                return inverse

            solved_y = torch.matmul(inverse, y)
            solved_Kcy = torch.matmul(inverse, phi_mat.mul(s_eig))
        else:
            Kyy = torch.matmul(phi_mat.mul(s_eig), phi_mat.T)
            Kyy = (Kyy + Kyy.T) / 2

            if nugget is not None:
                Kyy = Kyy + torch.eye(n) * nugget
            else:
                Kyy = Kyy

            if debug:
                return Kyy

            inverse = solver.inverse(
                Kyy
            )  # this is usually the Cholesky decomp - not the actual inverse
            solved_y = solver.solve(inverse, y.unsqueeze(1))
            solved_Kcy = solver.solve(inverse, phi_mat.mul(s_eig))

        mean = torch.matmul(phi_mat.mul(s_eig).T, solved_y)
        covariance_matrix = torch.diag(s_eig) - torch.matmul(
            phi_mat.mul(s_eig).T, solved_Kcy
        )
        covariance_matrix = (covariance_matrix + covariance_matrix.T) / 2

        return mean, covariance_matrix

    def condition_x(self, x, y, m, solver=None, nugget=None, debug=False):
        phi_mat = self.basis_matrix(x, m)
        return self.condition(phi_mat, y, solver, nugget, debug, m=m)

    def get_cov_matrices(
        self, phi_mat, nugget=None, inverse_only=False, solver=None, m=None
    ):
        """Compute the covariance matrix for the given phi_matrix .

        Args:
            phi_mat ([torch.Tensor]): [Matrix of basis function evaluations]
            nugget ([torch.Tensor], optional): [Nugget term that adds to diagonal of the covariance matrix]. Defaults to None.
            inverse_only (bool, optional): [description]. Defaults to False.
            solver ([type], optional): [description]. Defaults to None.
            m ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        if solver is None:
            solver = self.solver

        n, M = phi_mat.shape
        if m is None:
            m = int(M ** (1 / self.kernel.dim))

        s_eig = self.kernel.spectral_eig(m)
        Kyy = torch.matmul(phi_mat.mul(s_eig), phi_mat.T)
        Kyy = (Kyy + Kyy.T) / 2

        if nugget is not None:
            Kyy = Kyy + torch.eye(n) * nugget

        Kyy_inv = solver.inverse(Kyy)

        if inverse_only is True:
            return Kyy_inv

        Kcy = phi_mat.mul(s_eig).T
        solved_Kcy = solver.solve(Kyy_inv, phi_mat.mul(s_eig))

        covariance_matrix = torch.diag(s_eig) - torch.matmul(
            phi_mat.mul(s_eig).T, solved_Kcy
        )
        covariance_matrix = (covariance_matrix + covariance_matrix.T) / 2

        return covariance_matrix, Kyy_inv, Kcy

    def update_mean_vec(self, y, Kcy, Kyy_inv, vectorise=False, solver=None):
        """Compute the mean vector of YYY and update the mean vector .

        Args:
            y ([type]): [description]
            Kcy ([type]): [description]
            Kyy_inv ([type]): [description]
            vectorise (bool, optional): [description]. Defaults to False.
            solver ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """

        if solver is None:
            solver = self.solver
        if vectorise is True:
            solved_y = solver.vec_solve(Kyy_inv, y)
        else:
            solved_y = solver.solve(Kyy_inv, unsqueeze(y, 1))

        mean = torch.matmul(Kcy, solved_y)
        return mean

    def sample_coefficients(
        self, mean, covariance, n, random_sample=None, solver=None, sqrt=None
    ):

        if sqrt is None:
            if solver is None:
                sqrt = self.solver.square_root(
                    covariance
                )  # square root of c covariance matrix
            else:
                sqrt = solver.square_root(covariance)

        if random_sample is not None:
            sn_samples = random_sample
        else:
            sn = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
            sn_samples = sn.sample(mean.size() + torch.Size([n])).squeeze()

        mn_samples = torch.matmul(sqrt, sn_samples) + mean.unsqueeze(-1)

        return mn_samples

    def sample_mesh(
        self,
        mean,
        covariance,
        mesh,
        n,
        random_sample=None,
        solver=None,
        sqrt=None,
        m=None,
    ):
        dim = self.kernel.dim

        if m is None:
            m = int(mean.size(-1) ** (1 / dim))

        mn_samples = self.sample_coefficients(
            mean, covariance, n, random_sample, solver, sqrt
        )

        gram_mesh = self.basis_matrix(mesh, m)
        samples = torch.matmul(gram_mesh, mn_samples)

        return samples

    def sample(
        self, mean, covariance, n, random_sample=None, solver=None, sqrt=None, m=None
    ):

        dim = self.kernel.dim

        if m is None:
            M = mean.size(-1)
            m = int(M ** (1 / dim))

        mn_samples = self.sample_coefficients(
            mean, covariance, n, random_sample, solver, sqrt
        )

        def sampled_function(x):
            basis_eval = self.basis_matrix(x, m)
            return torch.matmul(basis_eval, mn_samples)

        return sampled_function

    def cross_covariance(self, x1, x2, m, parameters=None):
        """Compute the cross-covariance matrix at input locations x1 and x2

        Args:
            x1 ([torch.Tensor]): [First Tensor of input locations]
            x2 ([torch.Tensor]): [Second Tensor of input locations]
            m ([int, list]): [Number of basis functions]
            parameters ([list], optional): [Hyperparameters to use in computation]. Defaults to None.

        Returns:
            [type]: [description]
        """
        s_eig = self.kernel.spectral_eig(m, parameters)
        gram_mat_x1 = self.basis_matrix(x1, m).mul(s_eig)
        gram_mat_x2 = self.basis_matrix(x2, m)
        return torch.mm(gram_mat_x1, gram_mat_x2.t())

    def basis_matrix(self, x, m):
        """Compute the basis matrix at given locations x.

        Args:
            x ([torch.Tensor]): [Input locations]
            m ([list, int]): [Number of basis functions]

        Returns:
            [torch.Tensor]: [Matrix of basis function evaluations]
        """
        return self.kernel.basis_eval(x, m)

    def sample_prior(self, n, m, random_sample=None):
        """Sample from the prior GP

        Args:
            n ([int]): [Number of samples]
            m ([list, int]): [Number of basis functions]
            random_sample ([torch.Tensor], optional): [An optional Tensor of samples from the standard Gaussian]. Defaults to None.

        Returns:
            [function]: [Prior samples]
        """

        M = self.kernel.get_n_basis_funcs(m)

        s_eig = self.kernel.spectral_eig(m)
        s_cov = torch.diag(s_eig).sqrt()

        if random_sample is not None:
            sn_samples = random_sample
        else:
            sn = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
            sn_samples = sn.sample(torch.Size([M]) + torch.Size([n])).squeeze()

        mn_samples = torch.matmul(
            s_cov, sn_samples
        )  # need to add mean here if there is prior mean?

        def sampled_function(x):
            basis_eval = self.basis_matrix(x, m)
            return torch.matmul(basis_eval, mn_samples)

        return sampled_function

    def matheron_sample(
        self,
        phi_mat,
        y,
        n,
        random_sample=None,
        solver=None,
        nugget=None,
        inverse=None,
        m=None,
    ):
        """Sample from the posterior GP, using a given phi_mat and output observations y using the Matheron update approach.

        Args:
            phi_mat ([torch.Tensor]): [Matrix of basis function evaluations]
            y ([torch.Tensor]): [Output observations]
            n ([int]): [Number of posterior GP samples]
            random_sample ([torch.Tensor], optional): [An optional Tensor of samples from the standard Gaussian]. Defaults to None.
            solver ([Solver], optional): [Linear algebra solver to use]. Defaults to None.
            nugget ([torch.Tensor], optional): [Nugget term that adds to diagonal of the covariance matrix]. Defaults to None.
            inverse ([torch.Tensor], optional): [Optional inverse of covariance matrix]. Defaults to None.
            m ([list, int], optional): [Number of basis functions]. Defaults to None.

        Returns:
            [function]: [Posterior samples]
        """

        if solver is None:
            solver = self.solver

        N, M = phi_mat.shape
        if m is None:
            m = int(M ** (1 / self.kernel.dim))

        if y.dim() > 1:
            y_Size = torch.Size([y.shape[0]])
        else:
            y_Size = torch.Size([])

        s_eig = self.kernel.spectral_eig(m)
        s_cov = torch.sqrt(torch.diag(s_eig))

        if inverse is None:
            Kyy = torch.matmul(phi_mat.mul(s_eig), phi_mat.T)
            Kyy = (Kyy + Kyy.T) / 2
            if nugget is not None:
                Kyy = Kyy + torch.eye(N) * nugget
            inverse = solver.inverse(Kyy)

        if random_sample is not None:
            sn_samples = random_sample
        else:
            sn = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
            sn_samples = sn.sample(y_Size + torch.Size([M]) + torch.Size([n])).squeeze()

        mn_samples = torch.matmul(
            s_cov, sn_samples
        )  # need to add mean here if there is prior mean?
        matheron_solved = solver.vec_solve(
            inverse, y.unsqueeze(-1) - torch.matmul(phi_mat, mn_samples)
        )

        def sampled_function(x):
            basis_eval = self.basis_matrix(x, m)
            t1 = torch.matmul(basis_eval, mn_samples)
            t2 = torch.matmul(
                torch.matmul(basis_eval.mul(s_eig), phi_mat.T), matheron_solved
            )
            return t1 + t2

        return sampled_function

    def matheron_sample_mesh(
        self,
        phi_mat,
        y,
        mesh,
        n,
        random_sample=None,
        nugget=None,
        solver=None,
        inverse=None,
        m=None,
    ):
        """Sample from the posterior GP at given locations mesh, using phi_mat and output observations y and the Matheron update approach.

        Args:
            phi_mat ([torch.Tensor]): [Matrix of basis function evaluations]
            y ([torch.Tensor]): [Output observations]
            n ([int]): [Number of posterior GP samples]
            random_sample ([torch.Tensor], optional): [An optional Tensor of samples from the standard Gaussian]. Defaults to None.
            solver ([Solver], optional): [Linear algebra solver to use]. Defaults to None.
            nugget ([torch.Tensor], optional): [Nugget term that adds to diagonal of the covariance matrix]. Defaults to None.
            inverse ([torch.Tensor], optional): [Optional inverse of covariance matrix]. Defaults to None.
            m ([list, int], optional): [Number of basis functions]. Defaults to None.

        Returns:
            [torch.Tensor]: [Posterior samples evaluated at mesh]
        """

        if solver is None:
            solver = self.solver

        N, M = phi_mat.shape
        if m is None:
            m = int(M ** (1 / self.kernel.dim))

        s_eig = self.kernel.spectral_eig(m)
        s_cov = torch.sqrt(torch.diag(s_eig))

        if inverse is None:
            Kyy = torch.matmul(phi_mat.mul(s_eig), phi_mat.T)
            Kyy = (Kyy + Kyy.T) / 2
            if nugget is not None:
                Kyy = Kyy + torch.eye(N) * nugget
            inverse = solver.inverse(Kyy)

        if random_sample is not None:
            sn_samples = random_sample
        else:
            sn = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
            sn_samples = sn.sample(torch.Size([M]) + torch.Size([n])).squeeze()

        mn_samples = torch.matmul(
            s_cov, sn_samples
        )  # need to add mean here if there is prior mean?

        matheron_solved = solver.vec_solve(
            inverse, y.unsqueeze(-1) - torch.matmul(phi_mat, mn_samples)
        )

        basis_eval = self.basis_matrix(mesh, m)
        t1 = torch.matmul(basis_eval, mn_samples)
        t2 = torch.matmul(
            torch.matmul(basis_eval.mul(s_eig), phi_mat.T), matheron_solved
        )
        return t1 + t2

    def get_mean(self, mean, m=None):
        """Constructs the posterior mean function from given posterior mean vector.

        Args:
            mean ([torch.Tensor]): [Posterior mean Tensor]
            m ([list, int], optional): [Number of basis functions]. Defaults to None.

        Returns:
            [function]: [Posterior mean function]
        """

        dim = self.kernel.dim

        if m is None:
            m = round(mean.size(0) ** (1 / dim))

        def mean_func(x):
            basis_eval = self.basis_matrix(x, m)
            return torch.matmul(basis_eval, mean)

        return mean_func

    def get_variance(self, covariance_matrix, m=None):
        """Constructs the posterior pointwise variance function from given posterior covariance matrix.

        Args:
            covariance_matrix ([torch.Tensor]): [Posterior covariance matrix]
            m ([list, int], optional): [Number of basis functions]. Defaults to None.

        Returns:
            [function]: [Posterior pointwise variance function function]
        """

        dim = self.kernel.dim

        if m is None:
            m = round(covariance_matrix.size(0) ** (1 / dim))

        def variance_func(x):
            basis_eval = self.basis_matrix(x, m)
            return torch.clamp(torch.sum(
                torch.matmul(basis_eval, covariance_matrix) * basis_eval, dim=1
            ), min=0)

        return variance_func

    def set_amplitude_mle(self, phi_mat, y):  #TODO: Doesn't work
        """Sets the amplitude parameter of the kernel using closed MLE. 

        Args:
            phi_mat ([torch.Tensor]): [Matrix of basis function evaluations]
            y ([torch.Tensor]): [Output observations]
        """
        n, m = phi_mat.shape
        M = int(m ** (1 / self.kernel.dim))

        kernel_params = self.kernel.parameters

        s_eig = self.kernel.spectral_eig(
            M, parameters=[torch.Tensor([1])] + kernel_params[1:]
        )
        Kyy = torch.mm(phi_mat.mul(s_eig), phi_mat.T)
        Kyy = (Kyy + Kyy.T) / 2

        inverse = self.solver.inverse(Kyy)
        solved_y = self.solver.solve(inverse, unsqueeze(y, 1))
        exponent = torch.dot(y, solved_y)

        self.kernel.set_amplitude(torch.sqrt(exponent / n))

    def set_kernel_parameters(self, parameters):
        """Set the parameters of the kernel.

        Args:
            parameters ([list]): [List of kernel parameters]
        """
        self.kernel.set_parameters(parameters)

    def set_domain(self, domain):
        """Set the domain of the GP. For example, set_domain([[-1,1],[0,1]]) sets the domain of the GP as the set of points (x,y) satisfying -1 < x < 1, 0 < y < 1.

        Args:
            domain ([list, torch.Tensor]): [List or Tensor defining domain of GP]
        """

        if type(domain) is list:
            domain_t = torch.Tensor(domain)
        else:
            domain_t = domain

        self.domain = domain_t
        self.mean.set_domain(domain_t)
        self.kernel.set_domain(domain_t)

    def _check_dim(self, x):
        """Check that the dimension of x matches the dimension of the kernel.

        Args:
            x ([torch.Tensor]): [Tensor to check the dimension of the kernel against]

        Raises:
            ValueError: [Raises error if dimensions don't match]
        """
        x_dim = x.shape[1]
        if x_dim != self.kernel.dim:
            raise ValueError("Dimension of input doesn't match dimension of kernel.")


class TSpectralGP(SpectralGP):
    pass
