from GaussED.distribution.base import Distribution


class Gaussian(Distribution):  # TODO: implement standard GP

    def __init__(self, mean, kernel, exact=True):
        super().__init__()
        self.mean = mean
        self.kernel = kernel

        self._exact = exact  # Condition exactly or not

        self.func_dist = True  # distribution over functions
        self.func_dim = self.kernel.dim

    def forward(self, x):
        raise NotImplementedError
