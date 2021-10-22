import torch

from gaussed.solver.integral_transform.base import IntegrateSolver


class RiemannSum(IntegrateSolver):
    def __init__(self, mesh, volume):
        self.mesh = mesh
        self.volume = volume

    def solve(self, func):
        """Uses the Riemann sum approximation to compute the integral of func.

        Args:
            func ([function]): [Possibly vectorised function to integrate over]

        Returns:
            [torch.Tensor]: [Integral of func]
        """
        return self.volume * torch.mean(func(self.mesh))

    def line_integral_basis(self, func, m):
        """Uses the Riemann sum approximation to compute the integral of func over a flattened mesh.

        Args:
            func ([function]): [Function to integrate over]
            m ([int, list]): [Number of basis functions]

        Returns:
            [torch.Tensor]: [Integral of func]
        """
        return self.volume * torch.mean(func(self.mesh, m).T, dim=1)
