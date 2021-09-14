import torch
from torch import unsqueeze
from torch.distributions import MultivariateNormal

import math

import GaussED.transforms.linear as linear
from GaussED.distribution.gp import GP
from GaussED.mean.base import ZeroMean
from GaussED.utils.lin_alg_solvers import DefaultSolver
from GaussED.utils.summation_tensor_gen import sum_tensor_gen


class HilbertGP(GP):
    # keep a copy of the ids of all class instances using a class variable, each list in this list will be a list of
    # all GPs that are transformations of each other.
    t_ids = []

    def __init__(self, kernel, mean=None, solver=DefaultSolver(), transform=False, add_id=False):

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
        for i in HilbertGP.t_ids:
            for j in i:
                if id(self) is j:
                    return
        if transform is False:
            HilbertGP.t_ids.append([id(self)])

    def log_likelihood(self, phi_mat, y, solver=None, nugget=None, parameters=None, m=None):
        n, M = phi_mat.shape

        if m is None:
            m = int(M ** (1 / self.kernel.dim))

        if solver is None:
            solver = self.solver

        if parameters is None:
            parameters = self.kernel.parameters

        s_eig = self.kernel.spectral_eig(m, parameters)
        Kyy = torch.matmul(phi_mat.mul(s_eig), phi_mat.T)
        Kyy = (Kyy + Kyy.T) / 2

        if nugget is not None:
            Kyy = Kyy + torch.eye(n) * nugget
        else:
            Kyy = Kyy

        if n > M and nugget is not None:  # Woodbury matrix identity
            term = solver.inverse(torch.diag(s_eig.pow(-1)) + 1 / nugget * torch.matmul(phi_mat.T, phi_mat))
            solved = solver.solve(term, phi_mat.T)
            inverse = 1 / nugget * torch.eye(phi_mat.shape[0]) - 1 / (nugget ** 2) * torch.matmul(phi_mat, solved)
            solved_y = torch.matmul(inverse, y)
        else:
            inverse = solver.inverse(Kyy)  # this is usually the Cholesky decomp - not the actual inverse
            solved_y = solver.solve(inverse, y.unsqueeze(1))

        log_det_cov = solver.log_det(Kyy, inverse)
        exponent = torch.dot(y, solved_y)

        return -1 / 2 * (n * torch.log(torch.Tensor([2 * math.pi])) + log_det_cov + exponent)

    def log_likelihood_x(self, x, y, m, solver=None, nugget=None, parameters=None):
        phi_mat = self.basis_matrix(x, m)
        return self.log_likelihood(phi_mat, y, solver=solver, nugget=nugget, parameters=parameters)

    def condition(self, phi_mat, y, solver=None, nugget=None, debug=False, m=None):

        if solver is None:
            solver = self.solver

        n, M = phi_mat.shape
        if m is None:
            m = int(M ** (1 / self.kernel.dim))

        s_eig = self.kernel.spectral_eig(m)

        if n > M and nugget is not None:  # Woodbury matrix identity
            term = solver.inverse(torch.diag(s_eig.pow(-1)) + 1 / nugget * torch.matmul(phi_mat.T, phi_mat))
            solved = solver.solve(term, phi_mat.T)
            inverse = 1 / nugget * torch.eye(phi_mat.shape[0]) - 1 / (nugget ** 2) * torch.matmul(phi_mat, solved)

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

            inverse = solver.inverse(Kyy)  # this is usually the Cholesky decomp - not the actual inverse
            solved_y = solver.solve(inverse, y.unsqueeze(1))
            solved_Kcy = solver.solve(inverse, phi_mat.mul(s_eig))

        mean = torch.matmul(phi_mat.mul(s_eig).T, solved_y)
        covariance_matrix = torch.diag(s_eig) - torch.matmul(phi_mat.mul(s_eig).T, solved_Kcy)
        covariance_matrix = (covariance_matrix + covariance_matrix.T) / 2

        return mean, covariance_matrix

    def condition_x(self, x, y, m, solver=None, nugget=None, debug=False):
        phi_mat = self.basis_matrix(x, m)
        return self.condition(phi_mat, y, solver, nugget, debug)

    def get_cov_matrices(self, phi_mat, nugget=None, inverse_only=False, solver=None, m=None):
        """
        Function that computes the covariance matrices $K_{CY}$ and $K_{YY}$, such that
        $(K_{CY})_{ij} = \mathbb{C}(c_i,y_j)$ and $(K_{YY})_{ij} = \mathbb{C}(y_i,y_j)$, where $c_i$ is the $i$th basis
        coefficient and $y_j$ is the $j$th data.
        :param phi_mat: list with each element [gp object, x data, y data]
        :param nugget: number of basis functions in each dimension
        :param inverse_only:
        :param solver:
        :return: covariance_matrix, Kyy_inv, Kcy
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

        covariance_matrix = torch.diag(s_eig) - torch.matmul(phi_mat.mul(s_eig).T, solved_Kcy)
        covariance_matrix = (covariance_matrix + covariance_matrix.T) / 2

        return covariance_matrix, Kyy_inv, Kcy

    def update_mean_vec(self, y, Kcy, Kyy_inv, vectorise=False, solver=None):
        """
        Updates the posterior mean of $C = (c_1,\ldots, c_{m^{dim}})$.
        :param y: vector of data points
        :param Kcy:
        :param vectorise:
        :param Kyy_inv: "inverse" of covariance matrix Kyy - self.solver.inverse(Kyy)
        :param solver:
        :return: mean_vector - updated posterior mean vector
        """

        if solver is None:
            solver = self.solver
        if vectorise is True:
            solved_y = solver.vec_solve(Kyy_inv, y)
        else:
            solved_y = solver.solve(Kyy_inv, unsqueeze(y, 1))

        mean = torch.matmul(Kcy, solved_y)
        return mean

    def sample_coefficients(self, mean, covariance, n, random_sample=None, solver=None, sqrt=None):

        if sqrt is None:
            if solver is None:
                sqrt = self.solver.square_root(covariance)  # square root of c covariance matrix
            else:
                sqrt = solver.square_root(covariance)

        if random_sample is not None:
            sn_samples = random_sample
        else:
            sn = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
            sn_samples = sn.sample(mean.size() + torch.Size([n])).squeeze()

        mn_samples = torch.matmul(sqrt, sn_samples) + mean.unsqueeze(-1)

        return mn_samples

    def sample_mesh(self, mean, covariance, mesh, n, random_sample=None, solver=None, sqrt=None):
        dim = self.kernel.dim
        M = mean.size(-1)
        m = int(M ** (1 / dim))

        mn_samples = self.sample_coefficients(mean, covariance, n, random_sample, solver, sqrt)

        gram_mesh = self.basis_matrix(mesh, m)
        samples = torch.matmul(gram_mesh, mn_samples)

        return samples

    def sample(self, mean, covariance, n, random_sample=None, solver=None, sqrt=None, m=None):

        dim = self.kernel.dim

        if m is None:
            M = mean.size(-1)
            m = int(M ** (1 / dim))

        mn_samples = self.sample_coefficients(mean, covariance, n, random_sample, solver, sqrt)

        def sampled_function(x):
            basis_eval = self.basis_matrix(x, m)
            return torch.matmul(basis_eval, mn_samples)

        return sampled_function

    def cross_covariance(self, x1, x2, m, parameters=None):
        s_eig = self.kernel.spectral_eig(m, parameters)
        gram_mat_x1 = self.basis_matrix(x1, m).mul(s_eig)
        gram_mat_x2 = self.basis_matrix(x2, m)
        return torch.mm(gram_mat_x1, gram_mat_x2.t())

    def basis_matrix(self, x, m):
        return self.kernel.basis_eval(x, m)

    def sample_prior(self, n, m, random_sample=None):

        M = self.kernel.get_n_basis_funcs(m)

        s_eig = self.kernel.spectral_eig(m)
        s_cov = torch.diag(s_eig).sqrt()

        if random_sample is not None:
            sn_samples = random_sample
        else:
            sn = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
            sn_samples = sn.sample(torch.Size([M]) + torch.Size([n])).squeeze()

        mn_samples = torch.matmul(s_cov, sn_samples)  # need to add mean here if there is prior mean?

        def sampled_function(x):
            basis_eval = self.basis_matrix(x, m)
            return torch.matmul(basis_eval, mn_samples)

        return sampled_function

    def matheron_sample(self, phi_mat, y, n, random_sample=None, solver=None, nugget=None, inverse=None, m=None):

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

        mn_samples = torch.matmul(s_cov, sn_samples)  # need to add mean here if there is prior mean?
        matheron_solved = solver.vec_solve(inverse, y.unsqueeze(-1) - torch.matmul(phi_mat, mn_samples))

        def sampled_function(x):
            basis_eval = self.basis_matrix(x, m)
            t1 = torch.matmul(basis_eval, mn_samples)
            t2 = torch.matmul(torch.matmul(basis_eval.mul(s_eig), phi_mat.T), matheron_solved)
            return t1 + t2

        return sampled_function

    def matheron_sample_mesh(self, phi_mat, y, mesh, n, random_sample=None, nugget=None, solver=None, inverse=None):

        if solver is None:
            solver = self.solver

        N, M = phi_mat.shape
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

        mn_samples = torch.matmul(s_cov, sn_samples)  # need to add mean here if there is prior mean?

        matheron_solved = solver.vec_solve(inverse, y.unsqueeze(-1) - torch.matmul(phi_mat, mn_samples))

        basis_eval = self.basis_matrix(mesh, m)
        t1 = torch.matmul(basis_eval, mn_samples)
        t2 = torch.matmul(torch.matmul(basis_eval.mul(s_eig), phi_mat.T), matheron_solved)
        return t1 + t2


    def get_mean(self, mean):
        """
        Returns mean function or (if available) scalar value
        :param mean: mean of c vector
        :return: mean function
        """

        dim = self.kernel.dim

        M = round(mean.size(0) ** (1 / dim))

        def mean_func(x):
            basis_eval = self.basis_matrix(x, M)
            return torch.matmul(basis_eval, mean)

        return mean_func

    def set_amplitude_mle(self, phi_mat, y):  # TODO: make work
        n, m = phi_mat.shape
        M = int(m ** (1 / self.kernel.dim))

        kernel_params = self.kernel.parameters

        s_eig = self.kernel.spectral_eig(M, parameters=[torch.Tensor([1])] + kernel_params[1:])
        Kyy = torch.mm(phi_mat.mul(s_eig), phi_mat.T)
        Kyy = (Kyy + Kyy.T) / 2

        inverse = self.solver.inverse(Kyy)
        solved_y = self.solver.solve(inverse, unsqueeze(y, 1))
        exponent = torch.dot(y, solved_y)

        self.kernel.set_amplitude(torch.sqrt(exponent / n))

    def set_kernel_parameters(self, parameters):
        self.kernel.set_parameters(parameters)

    def set_domain(self, domain):
        self.domain = domain
        self.mean.set_domain(domain)
        self.kernel.set_domain(domain)

    def _check_dim(self, x):
        x_dim = x.shape[1]
        if x_dim != self.kernel.dim:
            raise ValueError("Dimension of input doesn't match dimension of kernel.")


class THilbertGP(HilbertGP):
    pass
