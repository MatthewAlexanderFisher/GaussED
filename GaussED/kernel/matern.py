import torch
import math

from GaussED.kernel.base import Kernel
from GaussED.basis.laplace import Laplace

default_params = [torch.Tensor([1]).requires_grad_(True), torch.Tensor([1]).requires_grad_(True)]

class MaternKernel(Kernel):

    def __init__(self, nu, dim=1, parameters=default_params, basis_class=None):
        super().__init__(dim, parameters, basis_class)
        self.nu = torch.Tensor([nu])

    def eval(self, x1, x2, parameters=None):

        if parameters is None:
            parameters = self.parameters

        amplitude, l_scale = parameters[:2]

        distance = torch.cdist(x1, x2, 2) / l_scale
        exp_component = torch.exp(-distance)

        if self.nu == torch.Tensor([0.5]):
            constant = 1
        elif self.nu == torch.Tensor([1.5]):
            constant = 1 + distance
        elif self.nu == torch.Tensor([2.5]):
            constant = 1 + distance + distance.pow(2)/3   #(math.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance ** 2)
        else:
            raise NotImplementedError("Only implemented for nu = 0.5, 1.5, 2.5")

        return amplitude ** 2 * constant * exp_component

    def __call__(self, distance, parameters=None):

        if parameters is None:
            parameters = self.parameters

        amplitude, l_scale = parameters[:2]

        exp_component = torch.exp(-distance)

        if self.nu == torch.Tensor([0.5]):
            constant = 1
        elif self.nu == torch.Tensor([1.5]):
            constant = 1 + distance
        elif self.nu == torch.Tensor([2.5]):
            constant = 1 + distance + distance.pow(2)/3   #(math.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance ** 2)
        else:
            raise NotImplementedError("Only implemented for nu = 0.5, 1.5, 2.5")

        return torch.exp(amplitude) * constant * exp_component

    def spectral_density(self, s, parameters=None):

        """
        Computes the spectral density with 1D input.
        :param s: Tensor of size 'n'
        :param parameters: list of hyperparameters of matern kernel - amplitude and lengthscale
        :param dim: Integer. Dimension of the domain of the kernel.
        :return: Tensor of size 'n'
        """

        if parameters is None:
            parameters = self.parameters

        amplitude, l_scale = parameters[:2]

        constant = torch.exp(amplitude) * 2 ** self.dim * math.pi ** (self.dim / 2) * (2 * self.nu) ** self.nu * \
                   torch.exp(torch.lgamma(self.nu + self.dim / 2)) / \
                   (torch.exp(torch.lgamma(self.nu)) * l_scale.pow(2 * self.nu))
        exp_component = s.pow(2) + (2 * self.nu) / (l_scale.pow(2))
        exponent = -(self.nu + self.dim / 2)

        return constant * exp_component.pow(exponent)

    def set_nu(self, nu):
        self.nu = nu
