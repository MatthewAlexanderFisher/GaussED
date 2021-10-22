from gaussed.transforms.base import Transform
import gaussed.distribution.spectralGP as hgp


class NonLinear(Transform):
    def __init__(self, distribution, method):
        super().__init__(distribution)
        self.linear = False
        self.method = method

    def __call__(self):
        t_GP = hgp.TSpectralGP(
            self._distribution.kernel, solver=self._distribution.solver, transform=True
        )
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

class OutputWarp(NonLinear):
    def __init__(self, distribution, func, bijective=None):

        self._distribution = distribution
        self.func = func

        if bijective is None:
            self.bijective = False
        else:
            self.bijective = True

        # def solve(samples):
            
        #     def f_samples(x):
        #         return func(samples(x))
            
        #     return f_samples

        # self.solve = solve

    def __call__(self):
        t_GP = hgp.TSpectralGP(
            self._distribution.kernel, solver=self._distribution.solver, transform=True
        )
        t_GP.domain = self._distribution.domain
        t_GP.set_kernel_parameters(self._distribution.kernel.parameters)

        def sample(*args, **kwargs):
            samples = self._distribution.sample(*args, **kwargs)
            return self.solve(samples)

        def sample_prior(*args, **kwargs):
            samples = self._distribution.sample_prior(*args, **kwargs)
            return self.solve(samples)

        def matheron_sample(*args, **kwargs):
            samples = self._distribution.matheron_sample(*args, **kwargs)
            return self.solve(samples)

        t_GP.sample = sample
        t_GP.sample_prior = sample_prior
        t_GP.matheron_sample = matheron_sample

        return t_GP

    def solve(self, samples):
        def f_samples(x):
            return self.func(samples(x))
        return f_samples