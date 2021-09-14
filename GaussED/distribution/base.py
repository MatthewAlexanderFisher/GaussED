

class Distribution:
    """
    Base class for distributions.
    """

    def forward(self):
        raise NotImplementedError

    def condition(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def log_likelihood(self):
        raise NotImplementedError

    def set_forward(self, forward_func):
        self.forward = forward_func

    def set_condition(self, condition_func):
        self.condition = condition_func

    def set_sample(self, sampling_func):
        self.sample = sampling_func
