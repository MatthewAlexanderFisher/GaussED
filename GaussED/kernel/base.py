import torch

from GaussED.utils.summation_tensor_gen import sum_tensor_gen
from GaussED.basis.laplace import Laplace


class Kernel:

    def __init__(self, dim, parameters, basis_class=None):
        self.dim = dim

        if basis_class is None:
            self.basis_class = Laplace
        else:
            self.basis_class = basis_class

        self.bases = [self.basis_class(self.dim)]
        self.parameters = parameters

        self.amplitude = parameters[0]

        self.stationary = None

    def eval(self, x1, x2, parameter=None):
        """
        Computes the covariance matrix between x1 and x2

        :param x1: Tensor 'n x d':
            First set of data.
        :param x2: Tensor 'm x d':
            Second set of data.
        :param parameter:
            :parameter list.
        :return: Tensor 'n x m':

        """
        raise NotImplementedError

    def basis_eval(self, x, m):
        return self.bases[0].eval(x, m)

    def spectral_density(self, x, parameter=None):
        """
        Computes the spectral density of the kernel function.

        :param x: Tensor of size 'n'
        :param parameter:
        :return: Tensor of size 'n'
        """
        raise NotImplementedError

    def spectral_eig(self, m, parameters=None):
        """
        Helper function for Laplace basis
        :param m: number of basis functions in each dimension
        :param parameters: the hyperparameters of the kernel
        :return: spectral density evaluated at eigenvalues of the Laplacian
        """

        dim = self.dim
        coeff = self.bases[0].b

        j_mat = sum_tensor_gen(dim, m)
        eig_val = torch.sum((j_mat * coeff).pow(2), dim=1)
        s_eig = self.spectral_density(torch.sqrt(eig_val), parameters)
        return s_eig

    def set_amplitude(self, amplitude):
        self.amplitude = amplitude
        self.parameters[0] = amplitude

    def set_parameters(self, parameters):
        self.parameters = parameters

    def set_domain(self, domain):
        for i in self.bases:
            i.set_domain(domain)

    def update_parameters(self, parameters):
        new_parameters = self.parameters + parameters
        self.parameters = new_parameters

    def get_n_basis_funcs(self, m):
        return int(m ** self.dim)


class CompositeKernel(Kernel):

    def __init__(self, *kernels):
        super().__init__(kernels[0].dim, kernels[0].parameters, kernels[0].basis_class)

        self.kernels = list(kernels)

        self.bases = []
        self.parameters = []  # a flattened version of bases_parameters_list
        self.bases_parameters_list = []  # a list of lists (ith element is parameter list of ith basis)

        for i in range(len(self.kernels)):
            self.bases_parameters_list.append(self.kernels[i].parameters)
            self.bases.append(self.kernels[i].bases[0])
            for j in range(len(self.kernels[i].parameters)):
                self.parameters.append(kernels[i].parameters[j])

    def eval(self, x1, x2, parameter=None):
        raise NotImplementedError

    def basis_eval(self, x, m):
        raise NotImplementedError

    def spectral_density(self, x, parameter=None):
        """
        Computes the spectral density of the kernel function.

        :param x: Tensor of size 'n'
        :param parameter:
        :return: Tensor of size 'n'
        """

        outputs = []
        for i in self.kernels:
            outputs.append[i.spectral_density(x)]

        return torch.stack(outputs)

    def spectral_eig(self, m_list, parameters=None):
        """
        Helper function for Laplace basis
        :param m_list: a list of m values for each kernel
        :param parameters: the hyperparameters of the kernel
        :return: spectral density evaluated at eigenvalues of the Laplacian
        """
        outputs = []

        for i in range(len(self.kernels)):
            dim = self.kernels[i].dim
            coeff = self.kernels[i].bases[0].b

            j_mat = sum_tensor_gen(dim, m_list[i])
            eig_val = torch.sum((j_mat * coeff).pow(2), dim=1)
            s_eig = self.kernels[i].spectral_density(torch.sqrt(eig_val), parameters)
            outputs.append(s_eig)

        return torch.cat(outputs)

    def set_parameters(self, parameters):
        self.parameters = parameters

    def update_parameters(self, parameters):
        new_parameters = self.parameters + parameters
        self.parameters = new_parameters

    def get_n_basis_funcs(self, m_list):
        M = 0
        for i in range(len(self.kernels)):
            M += int(m_list[i] ** self.kernels[i].dim)
        return M


class SumKernel(CompositeKernel):

    # TODO: the following doesn't work and needs to be more complex and handle the following cases:
    # (1) Where the GPs are unrelated and therefore have different coefficients.
    # (2) Where the GPs are related (i.e. transformations of each other) and have the same coefficients.
    # Currently basis_eval assumes the second case (2) and spectral_eig assumes the first case (1). We need to handle
    # both cases by introducing a related_kernel list of lists perhaps?

    def __init__(self, kernel_list, related_kernel_list):

        super().__init__(*kernel_list)

        self.related_kernel_list = related_kernel_list  # a list of lists

        self.bases = []
        self.parameters = []  # a flattened version of bases_parameters_list
        self.bases_parameters_list = []  # a list of lists (ith element is parameter list of ith basis)

        for i in range(len(self.kernels)):
            self.bases_parameters_list.append(self.kernels[i].parameters)
            self.bases.append(self.kernels[i].bases[0])
            for j in self.kernels[i].parameters:
                self.parameters.append[j]

    def eval(self, x1, x2, parameter=None):
        raise NotImplementedError

    def basis_eval(self, x, m):
        out = self.kernels[0].basis_eval(x, m)
        for j in range(len(self.kernels) - 1):
            out = out + self.kernels[j + 1].eval(x, m)
        return out

    def spectral_density(self, x, parameter=None):
        """
        Computes the spectral density of the kernel function.

        :param x: Tensor of size 'n'
        :param parameter:
        :return: Tensor of size 'n'
        """

        outputs = []
        for i in self.kernels:
            outputs.append[i.spectral_density(x)]

        return torch.stack(outputs)

    def spectral_eig(self, m, parameters=None):
        """
        Helper function for Laplace basis
        :param m:
        :param parameters: the hyperparameters of the kernel
        :return: spectral density evaluated at eigenvalues of the Laplacian
        """
        outputs = []

        for i in range(len(self.kernels)):
            dim = self.kernels[i].dim
            coeff = self.kernels[i].bases[0].b

            j_mat = sum_tensor_gen(dim, m)
            eig_val = torch.sum((j_mat * coeff).pow(2), dim=1)
            s_eig = self.kernels[i].spectral_density(torch.sqrt(eig_val), parameters)
            outputs.append(s_eig)
        return torch.cat(outputs)

    def set_parameters(self, parameters):
        self.parameters = parameters

    def update_parameters(self, parameters):
        new_parameters = self.parameters + parameters
        self.parameters = new_parameters
