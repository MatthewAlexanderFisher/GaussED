import torch

from GaussED.solver.optimise_transform.base import OptimiseSolver


class GridSearch(OptimiseSolver):

    def __init__(self, mesh):
        self.derivative = False
        self.mesh = mesh

    def solve(self, func, transpose=True):
        if transpose is True:
            return torch.max(func(self.mesh).T, dim=1)[0]
        else:
            return torch.max(func(self.mesh), dim=1)[0]

    def arg_max(self, func, transpose=True):
        if transpose is True:
            return self.mesh[torch.max(func(self.mesh).T, dim=1)[1]]
        else:
            return self.mesh[torch.max(func(self.mesh), dim=1)[1]]
