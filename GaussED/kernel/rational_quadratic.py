import torch
import math

from GaussED.kernel.base import Kernel
from GaussED.basis.laplace import Laplace

default_params = [torch.Tensor([1]).requires_grad_(True), torch.Tensor([1]).requires_grad_(True)]


class RationalQuadratic(Kernel):

    def __init__(self, shape=torch.Tensor([1]), dim=1, parameters=default_params, basis_class=None):
        super().__init__(dim, parameters, basis_class)

        self.shape = shape

    def eval(self, x1, x2, parameters=None):

        if parameters is None:
            parameters = self.parameters

        amplitude, l_scale = parameters[:2]

        distance = torch.cdist(x1, x2, 2).pow(2) / (2 * self.shape * l_scale.pow(2))

        return amplitude.pow(2) * (1 + distance).pow(-self.shape)

    def spectral_density(self, s, parameters=None):

        """
        Computes the spectral density of the squared exponential kernel for 1D positive input. The rational quadratic
        covariance function has the spectral density of a Gamma density, which we parameterise using the shape and scale
        parameters:
        $$ p(x; \alpha, l) = \frac{x^{\alpha - 1} e^{-x/l}}{(l^\alpha \gamma(\alpha))}$$
        :param s: Tensor of size 'n'
        :param parameters: list of hyperparameters of kernel - amplitude and lengthscale
        :return: Tensor of size 'n'
        """

        if parameters is None:
            parameters = self.parameters

        amplitude, l_scale = parameters[:2]

        const = amplitude.pow(2) / (l_scale.pow(self.shape) * torch.exp(torch.lgamma(self.shape)))

        return const * s.pow(self.shape - 1) * torch.exp(- s / l_scale)

    def set_shape(self, shape):
        self.shape = shape
