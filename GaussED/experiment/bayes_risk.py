import torch

from GaussED.experiment.acquisition import Acquisition
from GaussED.utils.lin_alg_solvers import SafeCholeskySolver


import matplotlib
from matplotlib import pyplot as plt
from matplotlib import style
style.use("ggplot")
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

class BayesRisk(Acquisition):

    def __init__(self, qoi, loss, design, solver=None, nugget=None):
        super().__init__(design)

        self.stochastic = True

        self.qoi = qoi
        self.loss = loss
        self.nugget = nugget

        if solver is None:
            self.solver = SafeCholeskySolver()
        else:
            self.solver = solver

        self.eval_params = {"m": 10, "n1": 81, "n2": 9}

        self.sn = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0]))

    def eval(self, d, phi_mat, y, m, n1=10, n2=10):

        phi_mat_new = torch.cat([phi_mat, self.design.basis_matrix(d.unsqueeze(0), m)])
        mean, cov = self.qoi.condition(phi_mat, y, solver=self.solver)

        sn_samples = self.sn.sample(mean.size() + torch.Size([n1])).squeeze()
        new_ys = self.design.sample(d, mean, cov, n1, random_sample=sn_samples, solver=self.solver)

        # get covariance matrix at new design
        cov_new, Kyy_inv, Kcy = self.qoi.get_cov_matrices(phi_mat_new, nugget=self.nugget, solver=self.solver)
        sqrt_cov_new = self.solver.square_root(cov_new)

        if not self.solver.vectorisable:
            total_loss = 0
            for i in range(n1):
                qoi_samp = self.qoi.sample(mean, cov, n1, sn_samples.T[i].T.unsqueeze(1), solver=self.solver)
                y_new_i = torch.cat([y, new_ys[i]])
                mean_new = self.qoi.update_mean_vec(y_new_i, Kcy, Kyy_inv, solver=self.solver)

                qoi_new_samp = self.qoi.sample(mean_new, None, n2, sqrt=sqrt_cov_new)

                total_loss = total_loss + torch.sum(self.loss(qoi_samp, qoi_new_samp))

            total_loss = 1 / (n1 * n2) * total_loss
        else:
            qoi_samps = self.qoi.sample(mean, cov, n1, sn_samples, solver=self.solver)
            all_ys = torch.cat([y.repeat(new_ys.size(0), 1).T, new_ys.T])
            mean_new = self.qoi.update_mean_vec(all_ys, Kcy, Kyy_inv, vectorise=True, solver=self.solver).T
            qoi_new_samps = self.qoi.sample(mean_new, None, n2, sqrt=sqrt_cov_new)
            total_loss = torch.mean(self.loss.vec_call(qoi_new_samps, qoi_samps))

        return total_loss

    def set_eval_params(self, eval_params):
        self.eval_params = eval_params

    def set_nugget(self, nugget):
        self.nugget = nugget

class BayesRiskMatheron(Acquisition):

    def __init__(self, qoi, loss, design, solver=None, nugget=None):
        super().__init__(design)

        self.stochastic = True

        self.qoi = qoi
        self.loss = loss
        self.nugget = nugget

        if solver is None:
            self.solver = SafeCholeskySolver()
        else:
            self.solver = solver

        self.eval_params = {"m": 10, "n1": 81, "n2": 9}

        self.sn = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0]))

    def eval(self, d, phi_mat, y, m, n1=10, n2=10):

        N, M = phi_mat.shape

        phi_mat_new = torch.cat([phi_mat, self.design.basis_matrix(d.unsqueeze(0), m)])

        Kyy_inv = self.qoi.get_cov_matrices(phi_mat, nugget=self.nugget, inverse_only=True, solver=self.solver)

        sn_samples = self.sn.sample(torch.Size([M]) + torch.Size([n1])).squeeze()
        new_ys = self.design.matheron_sample(d, phi_mat, y, n1, random_sample=sn_samples, solver=self.solver,
                                             nugget=self.nugget, inverse=Kyy_inv)

        # get covariance matrix at new design
        Kyy_inv_new = self.qoi.get_cov_matrices(phi_mat_new, nugget=self.nugget, inverse_only=True, solver=self.solver)

        if not self.solver.vectorisable:
            total_loss = 0

            for i in range(n1):
                qoi_samp = self.qoi.matheron_sample(phi_mat, y, None, random_sample=sn_samples.T[i].T.unsqueeze(1),
                                                    solver=self.solver, nugget=self.nugget, inverse=Kyy_inv)
                y_new_i = torch.cat([y, new_ys[i]])
                qoi_new_samp = self.qoi.matheron_sample(phi_mat_new, y_new_i, n2, solver=self.solver,
                                                        nugget=self.nugget, inverse=Kyy_inv_new)

                total_loss = total_loss + torch.sum(self.loss(qoi_samp, qoi_new_samp))

            total_loss = 1 / (n1 * n2) * total_loss
        else:
            qoi_samps = self.qoi.matheron_sample(phi_mat, y, n1, sn_samples, solver=self.solver, nugget=self.nugget,
                                                 inverse=Kyy_inv)
            all_ys = torch.cat([y.repeat(new_ys.size(0), 1).T, new_ys.T]).T
            qoi_new_samps = self.qoi.matheron_sample(phi_mat_new, all_ys, n2, solver=self.solver,
                                                     nugget=self.nugget, inverse=Kyy_inv_new)
            total_loss = torch.mean(self.loss.vec_call(qoi_new_samps, qoi_samps))

        return total_loss

    def set_eval_params(self, eval_params):
        self.eval_params = eval_params

    def set_nugget(self, nugget):
        self.nugget = nugget

