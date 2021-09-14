import torch

from GaussED.solver.integral_transform.base import IntegrateSolver


class RiemannSum(IntegrateSolver):

    def __init__(self, mesh, volume):
        self.mesh = mesh
        self.volume = volume

    def solve(self, func):
        return self.volume * torch.mean(func(self.mesh))

    def line_integral_basis(self, func, m):
        return self.volume * torch.mean(func(self.mesh, m).T, dim=1)
