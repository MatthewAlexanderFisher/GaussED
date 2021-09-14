import torch

class Optimiser:

    def __init__(self, objective, optim_method, variables, objective_params=None, optimiser_params=None):
        self.objective = objective
        self.optim_method = optim_method
        self.variables = variables

        if objective_params is None:
            self.objective_params = {}
        else:
            self.objective_params = objective_params

        if optimiser_params is None:
            self.optimiser_params = {}
        else:
            self.optimiser_params = optimiser_params

    def run(self, n, *args):
        pass

    def set_objective(self, objective):
        self.objective = objective

    def set_params(self, params):
        params.requires_grad_(True)
        self.params = params

    def set_func_params(self, params):
        self.func_params = params

    def set_optimiser_params(self, params):
        self.optimiser_params = params



class DefaultOptimiser(Optimiser):

    def __init__(self, objective, optim_method, variables, objective_params=None, optimiser_params=None):
        super().__init__(objective, optim_method, variables, objective_params, optimiser_params)

    def run(self, n, *args, retain_graph=False):
        optimiser = self.optim_method(self.variables, **self.optimiser_params)

        for i in range(n):
            optimiser.zero_grad()

            objective_eval = self.objective(*args, **self.objective_params)
            # print(self.objective_params)
            # if i % 10 == 0:
            #     print(objective_eval)
            objective_eval.backward(retain_graph=retain_graph)
            optimiser.step()

        return self.variables
