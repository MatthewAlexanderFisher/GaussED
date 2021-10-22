from gaussed.solver.optim.base import DefaultOptimiser
from gaussed.solver.optim.random_sample import SampleFirstOptimiser

import torch


class Experiment:
    def __init__(
        self,
        gp,
        black_box,
        design,
        acquisition,
        m,
        data=None,
        hyper_optim=None,
        acq_optim=None,
    ):
        self.gp = gp
        self.black_box = black_box
        self.design = design

        self.acquisition = acquisition
        self.acquisition.eval_params["m"] = m

        self.m = m

        self.domain = gp.domain

        if hyper_optim is None:

            def hyper_obj(a, b, solver=None, nugget=None):
                return -gp.log_likelihood(a, b, solver=solver, nugget=nugget, m=self.m)

            self.hyper_obj = hyper_obj

            hyper_obj_params = {
                "solver": self.acquisition.solver,
                "nugget": self.acquisition.nugget,
            }
            hyper_optim_params = {"lr": 1e-3}
            self.hyper_optim = DefaultOptimiser(
                self.hyper_obj,
                torch.optim.Adam,
                self.gp.kernel.parameters,
                objective_params=hyper_obj_params,
                optimiser_params=hyper_optim_params,
            )

        else:
            self.hyper_optim = hyper_optim

        if acq_optim is None:
            acq_obj = self.acquisition.eval
            acq_obj_params = self.acquisition.eval_params
            acq_optim_params = {"lr": 1e-1}
            self.acq_optim = SampleFirstOptimiser(
                acq_obj,
                torch.optim.Adam,
                self.design.sample_domain,
                objective_params=acq_obj_params,
                optimiser_params=acq_optim_params,
            )
        else:
            self.acq_optim = acq_optim

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
        """Run experiment for n iterations

        Args:
            n ([int]): [Number of iterations]
            optimise_hyper (bool, optional): [If True optimise the hyperparameters, otherwise don't]. Defaults to True.
            print_step (bool, optional): [If True print information at each iteration of running the experiment, otherwise don't]. Defaults to False.
            retain_graph (bool, optional): [If True pass True to retain_graph in the backward gradient pass, otherwise don't]. Defaults to False.
        """

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
                if print_step is True:
                    print("Finished hyperparmeter optimistion")

            current_phi_mat = self.design.basis_matrix(current_design, self.m)

            d = self.acq_optim.run(self.acq_optim_steps, *(current_phi_mat, current_data),
                                   retain_graph=retain_graph)

            if print_step is True:
                print("Finished acquisition optimistion")
                print("Design point: ", d)
            current_design = torch.cat([current_design, d.detach().unsqueeze(0)])
            current_data = torch.cat([current_data, self.black_box(d.detach())])
            current_phi_mat = self.design.update_basis_matrix(current_phi_mat, d.detach(), self.m)

            self.set_current_design(current_design)
            self.set_data(current_data)
            self.set_basis_mat(current_phi_mat)

    def set_basis_mat(self, basis_mat):
        """Sets the basis matrix

        Args:
            basis_mat ([torch.Tensor]): [Matrix of basis evaluations]
        """
        self.basis_mat = basis_mat

    def set_current_design(self, current_design):
        """Sets the current design.

        Args:
            current_design ([torch.Tensor]): [Tensor of design parameters that have been evaluated]
        """
        self.current_design = current_design

    def set_data(self, data):
        """Sets the output observations.

        Args:
            data ([torch.Tensor]): [Set the output observations]
        """
        self.data = data

    def set_hyperparameter_optimiser(self, optimiser):
        """Set the optimiser object for hyperparameter optimisation.

        Args:
            optimiser ([Optimiser]): [Optimiser object]
        """
        self.hyper_optim = optimiser

    def set_acquisition_optimiser(self, optimiser):
        """Set the optimiser object for acquisition optimisation.

        Args:
            optimiser ([Optimiser]): [Optimiser object]
        """
        self.acq_optim = optimiser

    def set_hyperparameter_optimiser_steps(self, n_steps):
        """Set number of steps to be used in the hyperparameter optimiser.

        Args:
            n_steps ([int]): [Number of optimisation steps]
        """
        self.hyper_optim_steps = n_steps

    def set_acquisition_optimiser_steps(self, n_steps):
        """Set number of steps to be used in the acquisition optimiser.

        Args:
            n_steps ([int]): [Number of optimisation steps]
        """
        self.hyper_optim_steps = n_steps

    def set_acquisition_params(self, acquisition_params):
        """Sets the dictionary of default parameters for the evaluation function of the acquisition object.

        Args:
            acquisition_params ([dict]): [Dictionary of parameters to be passed to the acquisition evaluation function]
        """
        self.acquisition.eval_params = acquisition_params
        self.m = acquisition_params.get("m")
