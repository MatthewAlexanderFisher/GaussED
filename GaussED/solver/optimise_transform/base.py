from GaussED.solver.base import Solver

class OptimiseSolver(Solver):

    def __init__(self):
        self.derivative = None

    def solve(self, x):
        raise NotImplementedError
