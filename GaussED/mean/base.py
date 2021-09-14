import torch

class Mean:

    def __init__(self, dim):
        self.dim = dim
        self.parameters = []

    def eval(self, x):
        raise NotImplementedError

    def set_domain(self, domain):
        pass


class ZeroMean(Mean):

    def __init__(self, dim):
        super().__init__(dim)
        self.parameters = []

    def eval(self, x):
        return 0

    def set_domain(self, domain):
        pass
