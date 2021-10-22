import torch
import math

from gaussed.kernel.base import Kernel

default_params = [
    torch.Tensor([1/2]).requires_grad_(True),
    torch.Tensor([0.2]).requires_grad_(True),
]


class MaternKernel(Kernel):
    def __init__(self, nu, dim=1, parameters=default_params, basis_class=None):
        super().__init__(dim, parameters, basis_class)
        self.nu = torch.Tensor([nu])

    def eval(self, x1, x2, parameters=None):

        """Returns the cross-covariance matrix of the kernel at given input locations.

        Raises:
            NotImplementedError: [If the smoothness parameter is not 0.5, 1.5 or 2.5, raise error.]

        Returns:
            [torch.Tensor]: [Cross-covariance matrix at given input locations.]
        """

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
            constant = (
                1 + distance + distance.pow(2) / 3
            )  # (math.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance ** 2)
        else:
            raise NotImplementedError("Only implemented for nu = 0.5, 1.5, 2.5")

        return torch.exp(amplitude) * constant * exp_component

    def spectral_density(self, s, parameters=None):

        """Returns the spectral density of the kernel at given 1D input locations.

        Returns:
            [torch.Tensor]: [Spectral density of the kernel]
        """

        if parameters is None:
            parameters = self.parameters

        amplitude, l_scale = parameters[:2]

        constant = (
            torch.exp(amplitude)
            * 2 ** self.dim
            * math.pi ** (self.dim / 2)
            * (2 * self.nu) ** self.nu
            * torch.exp(torch.lgamma(self.nu + self.dim / 2))
            / (torch.exp(torch.lgamma(self.nu)) * l_scale.pow(2 * self.nu))
        )
        exp_component = s.pow(2) + (2 * self.nu) / (l_scale.pow(2))
        exponent = -(self.nu + self.dim / 2)

        return constant * exp_component.pow(exponent)

    def set_nu(self, nu):
        """Sets the smoothness parameter of the kernel.

        Args:
            nu ([torch.Tensor]): [Smoothness parameter]
        """
        self.nu = nu
