from gaussed.solver.base import Solver


class IntegrateSolver(Solver):
    def __init__(self):
        self.derivative = None

    def solve(self, x):
        raise NotImplementedError
