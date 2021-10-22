import torch
import math

from gaussed.kernel.base import Kernel

default_params = [
    torch.Tensor([1]).requires_grad_(True),
    torch.Tensor([1]).requires_grad_(True),
]


class SquaredExponentialKernel(Kernel):
    def __init__(self, dim=1, parameters=default_params, basis_class=None):
        super().__init__(dim, parameters, basis_class)

    def eval(self, x1, x2, parameters=None):

        if parameters is None:
            parameters = self.parameters

        distance = torch.cdist(x1, x2, 2).pow(2)  # / l_scale.pow(2)

        return self.__call__(distance, parameters)

    def __call__(self, d, parameters=None):

        if parameters is None:
            parameters = self.parameters

        amplitude, l_scale = parameters[:2]

        return amplitude.pow(2) * torch.exp(-(d / l_scale).pow(2) / 2)

    def spectral_density(self, s, parameters=None):

        """
        Computes the spectral density of the squared exponential kernel with 1D input.
        :param s: Tensor of size 'n'
        :param parameters: list of hyperparameters of kernel - amplitude and lengthscale
        :return: Tensor of size 'n'
        """

        if parameters is None:
            parameters = self.parameters

        amplitude, l_scale = parameters

        const = amplitude.pow(2) * (2 * math.pi * l_scale.pow(2)).pow(1 / 2)
        # const = amplitude.pow(2) * (l_scale.pow(2) / math.pi).pow(1 / 2) / 2

        # return const * torch.exp(- 2 * math.pi ** 2 * l_scale.pow(2) * s.pow(2))
        return const * torch.exp(-(s * l_scale).pow(2) / 2)
