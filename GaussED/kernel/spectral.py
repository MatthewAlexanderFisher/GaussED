import torch
import math

from GaussED.kernel.base import Kernel
from GaussED.basis.laplace import Laplace

default_params = [torch.Tensor([1]).requires_grad_(True), torch.Tensor([1]).requires_grad_(True)]


class SpectralKernel(Kernel):

    """
    Learns the spectral representation of the kernel directly
    """

    def __init__(self, dim=1, parameters=default_params, basis_class=None):
        super().__init__(dim, parameters, basis_class)

    def eval(self, x1, x2, parameters=None):
        raise NotImplementedError()

    def spectral_density(self, s, parameters=None):
        raise NotImplementedError()