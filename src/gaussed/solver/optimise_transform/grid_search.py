import torch

from gaussed.solver.optimise_transform.base import OptimiseSolver


class GridSearch(OptimiseSolver):
    def __init__(self, mesh):
        self.derivative = False
        self.mesh = mesh

    def solve(self, func, transpose=True):
        """Implementation of grid search optimisation. Optimises func by selecting the maximal value of func over a given mesh.
        
        Args:
            func ([function]): [Possibly vectorised function to optimise over]
            transpose (bool, optional): [If True takes the transpose of the evaluations of func - required when func is vectorised]. Defaults to True.

        Returns:
            [torch.Tensor]: [Optimum obtained]
        """
        if transpose is True:
            return torch.max(func(self.mesh).T, dim=1)[0]
        else:
            return torch.max(func(self.mesh), dim=1)[0]

    def arg_max(self, func, transpose=True):
        """Implementation of grid search optimisation. Optimises func by selecting the maximal value of func over a given mesh and returns its argument.

        Args:
            func ([function]): [Possibly vectorised function to optimise over]
            transpose (bool, optional): [If True takes the transpose of the evaluations of func - required when func is vectorised]. Defaults to True.

        Returns:
            [torch.
        """
        if transpose is True:
            return self.mesh[torch.max(func(self.mesh).T, dim=1)[1]]
        else:
            return self.mesh[torch.max(func(self.mesh), dim=1)[1]]
