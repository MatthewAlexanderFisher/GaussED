from GaussED.transforms.base import Transform
from GaussED.distribution.base import Distribution
import GaussED.distribution.hilbertGP as hgp


class NonLinear(Transform):

    def __init__(self, distribution, method):
        super().__init__(distribution)
        self.linear = False
        self.method = method

    def __call__(self):
        t_GP = hgp.THilbertGP(self._distribution.kernel, solver=self._distribution.solver, transform=True)
        t_GP.domain = self._distribution.domain
        t_GP.set_kernel_parameters(self._distribution.kernel.parameters)

        def sample(*args, **kwargs):
            samples = self._distribution.sample(*args, **kwargs)
            return self.method.solve(samples)

        def sample_prior(*args, **kwargs):
            samples = self._distribution.sample_prior(*args, **kwargs)
            return self.method.solve(samples)

        def matheron_sample(*args, **kwargs):
            samples = self._distribution.matheron_sample(*args, **kwargs)
            return self.method.solve(samples)

        t_GP.sample = sample
        t_GP.sample_prior = sample_prior
        t_GP.matheron_sample = matheron_sample

        return t_GP


class Maximise(NonLinear):

    def __init__(self, distribution, method):
        super().__init__(distribution, method)
        # self.dims = dims  # dims is not used yet... TODO: do we need dims (e.g. will return functions)
    #
    # def __call__(self):
    #     t_GP = hgp.THilbertGP(self._distribution.kernel, solver=self._distribution.solver, transform=True)
    #     t_GP.domain = self._distribution.domain
    #     t_GP.set_kernel_parameters(self._distribution.kernel.parameters)
    #     t_GP.t_dim = 0
    #
    #     def sample(mean, covariance, n, random_sample=None, solver=None, sqrt=None):
    #         samples = self._distribution.sample(mean, covariance, n, random_sample=random_sample, solver=solver,
    #                                             sqrt=sqrt)
    #         return self.method.solve(samples)
    #
    #     def sample_prior(n, m, random_sample=None):
    #         samples = self._distribution.sample_prior(n, m, random_sample=random_sample)
    #         return self.method.solve(samples)
    #
    #     def matheron_sample(phi_mat, y, n, random_sample=None, solver=None, inverse=None):
    #         samples = self._distribution.matheron_sample(phi_mat, y, n, random_sample=None, solver=None, inverse=None)
    #         return self.method.solve(samples)
    #
    #     t_GP.sample = sample
    #     t_GP.sample_prior = sample_prior
    #     t_GP.matheron_sample = matheron_sample
    #
    #     return t_GP
