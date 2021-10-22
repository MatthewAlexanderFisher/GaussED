import torch
import math

from gaussed.kernel.base import Kernel
from gaussed.basis.laplace import Laplace

default_params = [
    torch.Tensor([1]).requires_grad_(True),
    torch.Tensor([1]).requires_grad_(True),
]


class RationalQuadratic(Kernel):
    def __init__(
        self,
        shape=torch.Tensor([1]),
        dim=1,
        parameters=default_params,
        basis_class=None,
    ):
        super().__init__(dim, parameters, basis_class)

        self.shape = shape

    def eval(self, x1, x2, parameters=None):

        """Returns the cross-covariance matrix of the kernel at given input locations.

        Returns:
            [torch.Tensor]: [Cross-covariance matrix at given input locations.]
        """

        if parameters is None:
            parameters = self.parameters

        amplitude, l_scale = parameters[:2]

        distance = torch.cdist(x1, x2, 2).pow(2) / (2 * self.shape * l_scale.pow(2))

        return amplitude.pow(2) * (1 + distance).pow(-self.shape)

    def spectral_density(self, s, parameters=None):

        """Returns the spectral density of the kernel at given 1D input locations.

        Returns:
            [torch.Tensor]: [Spectral density of the kernel.]
        """

        if parameters is None:
            parameters = self.parameters

        amplitude, l_scale = parameters[:2]

        const = amplitude.pow(2) / (
            l_scale.pow(self.shape) * torch.exp(torch.lgamma(self.shape))
        )

        return const * s.pow(self.shape - 1) * torch.exp(-s / l_scale)

    def set_shape(self, shape):
        """Set the shape parameter of the kernel.

        Args:
            shape ([torch.Tensor]): [The shape parameter]
        """
        self.shape = shape
