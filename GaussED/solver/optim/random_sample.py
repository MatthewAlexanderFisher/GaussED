from GaussED.solver.optim.base import Optimiser

from matplotlib import pyplot as plt

import torch

class SampleFirstOptimiser(Optimiser):

    """
    Attempts to find a good initialisation by evaluating the (noisy) function over a random sample of points over
    the domain of the function and then proceeds to perform stochastic optimisation from the best sample point.
    """

    def __init__(self, objective, optim_method, sampling_func, N=None, objective_params=None, optimiser_params=None):
        super().__init__(objective, optim_method, None, objective_params, optimiser_params)
        self.sampling_func = sampling_func

        if N is None:
            self.N = 100

        self.vectorisable_func = False

    def run(self, n, *args, retain_graph=False, debug=False):

        sample_N = self.sampling_func(self.N).detach()

        if debug is True:
            vals = []

        if self.vectorisable_func is True:
            best_sample = sample_N[torch.argmin(self.objective(sample_N, **self.objective_params))]
        else:
            best_sample = sample_N[0]
            best_val = self.objective(sample_N[0], *args, **self.objective_params)
            if debug is True:
                vals.append(best_val.unsqueeze(0).detach())
            for i in range(self.N - 1):
                f_samp = self.objective(sample_N[i+1], *args, **self.objective_params)
                if debug is True:
                    vals.append(f_samp.unsqueeze(0).detach())
                if f_samp < best_val:
                    best_val = f_samp
                    best_sample = sample_N[i+1]

        if debug is True:
            vals = torch.cat(vals)
            plt.scatter(sample_N.T[0], sample_N.T[1], c=vals, cmap=plt.cm.autumn)
            plt.colorbar()
            plt.show()
            print(best_val)
            print(best_sample)

        best_sample.requires_grad_(True)

        self.variables = [best_sample]

        optimiser = self.optim_method(self.variables, **self.optimiser_params)
        for i in range(n):
            optimiser.zero_grad()

            objective_eval = self.objective(self.variables[0], *args, **self.objective_params)
            objective_eval.backward(retain_graph=retain_graph)
            optimiser.step()

        return self.variables[0]
