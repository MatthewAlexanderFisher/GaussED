import torch


class Optimiser:
    def __init__(
        self,
        objective,
        optim_method,
        variables,
        objective_params=None,
        optimiser_params=None,
    ):
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

        self.optional_func = None

    def run(self, n, *args):
        """Runs n iterations of optimiser.

        Args:
            n ([int]): [Number of optimisation steps of optimiser]
        """

        pass

    def set_objective(self, objective):
        """Set objective function that will be optimised.

        Args:
            objective ([function]): [Objective function]
        """
        self.objective = objective

    def set_params(self, params):
        """Sets the default parameters of the objective function.

        Args:
            params ([dict]): [Dictionary of parameters to pass to objective function]
        """
        params.requires_grad_(True)
        self.params = params

    def set_optimiser_params(self, params):
        """Sets the default parameters of the optimiser.
        Args:
            params ([dict]): [Dictionary of parameters to pass to the optimiser]
        """
        self.optimiser_params = params

    def set_optional_func(self, optional_func):
        """Set the optional function that is used in the optimisation step.

        Args:
            optional_func ([function]): [Optional function]
        """
        self.optional_func = optional_func


class DefaultOptimiser(Optimiser):
    def __init__(
        self,
        objective,
        optim_method,
        variables,
        objective_params=None,
        optimiser_params=None,
    ):
        super().__init__(
            objective, optim_method, variables, objective_params, optimiser_params
        )

    def run(self, n, *args, retain_graph=False):
        """Run the optimizer for n iterations.

        Args:
            n ([int]): [Number of iterations of optimisation]
            retain_graph (bool, optional): [Passes to backward pass]. Defaults to False.

        Returns:
            [torch.Tensor]: [Optimised variables]
        """

        optimiser = self.optim_method(self.variables, **self.optimiser_params)

        for i in range(n):
            optimiser.zero_grad()

            objective_eval = self.objective(*args, **self.objective_params)
            objective_eval.backward(retain_graph=retain_graph)
            optimiser.step()

            if self.optional_func is not None:
                self.optional_func()

        return self.variables
