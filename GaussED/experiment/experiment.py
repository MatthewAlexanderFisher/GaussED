from GaussED.solver.optim.base import DefaultOptimiser
from GaussED.solver.optim.random_sample import SampleFirstOptimiser

import torch


class Experiment:

    def __init__(self, gp, black_box, design, acquisition, m, data=None, hyper_optim=None, acq_optim=None):
        self.gp = gp
        self.black_box = black_box
        self.design = design

        self.acquisition = acquisition
        self.acquisition.eval_params["m"] = m

        self.m = m

        self.domain = gp.domain

        if hyper_optim is None:

            def hyper_obj(a, b, solver=None, nugget=None):
                return - gp.log_likelihood(a, b, solver=solver, nugget=nugget)

            hyper_obj_params = {"solver": self.acquisition.solver, "nugget": self.acquisition.nugget}
            hyper_optim_params = {"lr": 1e-3}
            self.hyper_optim = DefaultOptimiser(hyper_obj, torch.optim.Adam, self.gp.kernel.parameters,
                                                objective_params=hyper_obj_params, optimiser_params=hyper_optim_params)
        if acq_optim is None:
            acq_obj = self.acquisition.eval
            acq_obj_params = self.acquisition.eval_params
            acq_optim_params = {"lr": 1e-1}
            self.acq_optim = SampleFirstOptimiser(acq_obj, torch.optim.Adam, self.design.sample_domain,
                                                  objective_params=acq_obj_params, optimiser_params=acq_optim_params)

        self.acq_optim_steps = 1000
        self.hyper_optim_steps = 1000
        self.start_hyp_optimising_step = -1

        self.current_design = design.initial_design
        self.basis_mat = self.design.basis_matrix(self.current_design, self.m)

        if data is None:
            self.data = self.black_box(self.current_design).flatten()
        else:
            self.data = data

    def run(self, n, optimise_hyper=True, print_step=False, retain_graph=False):

        current_design = self.current_design
        current_data = self.data
        current_phi_mat = self.design.basis_matrix(current_design, self.m)

        for i in range(n):
            if print_step is True:
                print("Step ", i+1, "/", n)

            if optimise_hyper is True:

                if i > self.start_hyp_optimising_step:
                    hyper_params = self.hyper_optim.run(self.hyper_optim_steps, *(current_phi_mat, current_data),
                                                        retain_graph=retain_graph)

            current_phi_mat = self.design.basis_matrix(current_design, self.m)

            d = self.acq_optim.run(self.acq_optim_steps, *(current_phi_mat, current_data),
                                   retain_graph=retain_graph)

            current_design = torch.cat([current_design, d.detach().unsqueeze(0)])
            current_data = torch.cat([current_data, self.black_box(d.detach())])
            current_phi_mat = self.design.update_basis_matrix(current_phi_mat, d.detach(), self.m)

            self.set_current_design(current_design)
            self.set_data(current_data)
            self.set_basis_mat(current_phi_mat)

    def set_basis_mat(self, basis_mat):
        self.basis_mat = basis_mat

    def set_current_design(self, current_design):
        self.current_design = current_design

    def set_data(self, data):
        self.data = data

    def set_hyperparameter_optimiser(self, optimiser):
        self.hyper_optim = optimiser

    def set_acquisition_optimiser(self, optimiser):
        self.acq_optim = optimiser

    def set_hyperparameter_optimiser_steps(self, n_steps):
        self.hyper_optim_steps = n_steps

    def set_acquisition_optimiser_steps(self, n_steps):
        self.hyper_optim_steps = n_steps

    def set_acquisition_params(self, acquisition_params):
        self.acquisition.eval_params = acquisition_params
        self.m = acquisition_params.get("m")


class ExperimentOld:

    def __init__(self, gp, black_box, design, acquisition, m, data=None):
        self.gp = gp
        self.black_box = black_box
        self.design = design

        self.acquisition = acquisition
        self.acquisition.eval_params["m"] = m

        self.m = m

        self.domain = gp.domain

        self.optimise_method = torch.optim.Adam
        self.optimiser_params = {"lr": 1e-1}
        self.optimiser_steps = 1000

        self.parameter_optimiser_params = {"lr": 1e-3}
        self.parameter_optimiser_steps = 1000
        self.parameter_optimiser = None

        self.current_design = design.initial_design
        self.basis_mat = self.design.basis_matrix(self.current_design, self.m)

        if data is None:
            self.data = self.black_box(self.current_design).flatten()
        else:
            self.data = data

    def run(self, n, optimise_kernel_params=True, print_step=False, debug=False, retain_graph=False):

        current_design = self.current_design
        current_data = self.data
        current_phi_mat = self.design.basis_matrix(current_design, self.m)

        for i in range(n):
            if print_step is True:
                print("Step ", i+1, "/", n)

            d = self.design.sample_domain().detach().requires_grad_(True)

            if debug:
                starting_point = self.design.transform(d.detach().clone())

            optimiser = self.optimise_method([d], **self.optimiser_params)

            for j in range(self.optimiser_steps):
                optimiser.zero_grad()

                out = self.acquisition.eval(d, current_phi_mat, current_data, **self.acquisition.eval_params)

                out.backward(retain_graph=retain_graph)
                optimiser.step()

            if optimise_kernel_params is True:

                if self.parameter_optimiser is None:
                    parameter_optimiser = self.optimise_method(self.gp.kernel.parameters, **self.parameter_optimiser_params)
                else:
                    parameter_optimiser = self.parameter_optimiser

                for j in range(self.parameter_optimiser_steps):
                    parameter_optimiser.zero_grad()
                    L = - self.gp.log_likelihood(current_phi_mat, current_data,
                                                 solver=self.acquisition.solver, nugget=self.acquisition.nugget)

                    L.backward()
                    parameter_optimiser.step()

            current_design = torch.cat([current_design, d.detach().unsqueeze(0)])
            current_data = torch.cat([current_data, self.black_box(d.detach())])
            current_phi_mat = self.design.update_basis_matrix(current_phi_mat, d.detach(), self.m)

            if debug:
                print("start: ", starting_point, ". end: ", self.design.transform(d.detach()))

            self.set_current_design(current_design)
            self.set_data(current_data)
            self.set_basis_mat(current_phi_mat)

    def set_basis_mat(self, basis_mat):
        self.basis_mat = basis_mat

    def set_current_design(self, current_design):
        self.current_design = current_design

    def set_data(self, data):
        self.data = data

    def set_optimiser_method(self, method):
        self.optimise_method = method

    def set_optimiser_steps(self, optimiser_steps):
        self.optimiser_steps = optimiser_steps

    def set_parameter_optimiser(self, parameter_optimiser):
        self.parameter_optimiser = parameter_optimiser

    def set_optimiser_params(self, optimiser_params):
        """
        Sets the optimiser parameters
        :param optimiser_params: a dictionary of input parameters to torch.optimiser
        """
        self.optimiser_params = optimiser_params

    def set_acquisition_params(self, acquisition_params):
        self.acquisition.eval_params = acquisition_params
        self.m = acquisition_params.get("m")

