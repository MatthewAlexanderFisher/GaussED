import torch

from GaussED.experiment.acquisition import Acquisition
from GaussED.utils.lin_alg_solvers import SafeCholeskySolver


class ExpectedImprovement(Acquisition):

    def __init__(self, gp, design, solver=SafeCholeskySolver(), nugget=None):
        super().__init__(design)

        self.gp = gp
        self.solver = solver
        self.nugget = nugget

        self.eval_params = {"m": 10, "n": 30}

    def eval(self, d, current_design, y, m, n=50):
        current_max = torch.max(y)

        phi_mat_current = self.design.basis_matrix(current_design, m)
        mean, cov = self.gp.condition(phi_mat_current, y, solver=self.solver)

        samples_d = torch.clamp(self.gp.sample_mesh(mean, cov, d.unsqueeze(1), n, solver=self.solver) - current_max, 0)

        return -torch.mean(samples_d, dim=1)

    def set_eval_params(self, eval_params):
        self.eval_params = eval_params

    def set_nugget(self, nugget):
        self.nugget = nugget
