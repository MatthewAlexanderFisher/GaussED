import torch
from copy import deepcopy
from functools import lru_cache

from GaussED.transforms.base import Transform
from GaussED.kernel.base import Kernel
import GaussED.distribution.hilbertGP as hgp


class Linear(Transform):

    def __init__(self, distribution):
        super().__init__(distribution)
        self.linear = True
        self.kernel = deepcopy(self._distribution.kernel)

    def forward(self, inputs):
        raise NotImplementedError

    def append_ids(self, dist):
        for i in range(len(hgp.HilbertGP.t_ids)):
            if id(self._distribution) in hgp.HilbertGP.t_ids[i]:
                hgp.HilbertGP.t_ids[i].append(id(dist))
                return

    def set_kernel_bases(self, new_bases):
        self.kernel.bases = new_bases


class Identity(Linear):

    def __init__(self, distribution):
        super().__init__(distribution)

    def __call__(self):
        t_GP = hgp.HilbertGP(self.kernel, solver=self._distribution.solver, transform=True)
        t_GP.domain = self._distribution.domain
        t_GP.set_kernel_parameters(self._distribution.kernel.parameters)
        self.append_ids(t_GP)
        return t_GP


class Differentiate(Linear):

    def __init__(self, distribution, diff_dims, orders):
        super().__init__(distribution)
        self.diff_dims = diff_dims
        self.orders = orders

    def __call__(self):
        t_bases = []
        for i in range(len(self.kernel.bases)):
            t_bases.append(self.kernel.bases[i].differentiate(self.kernel.bases[i], self.diff_dims, self.orders))

        self.set_kernel_bases(t_bases)

        t_GP = hgp.HilbertGP(self.kernel, solver=self._distribution.solver, transform=True)
        t_GP.domain = self._distribution.domain
        t_GP.set_kernel_parameters(self._distribution.kernel.parameters)
        self.append_ids(t_GP)

        return t_GP


class Integrate(Linear):

    def __init__(self, distribution, dims, domain):
        super().__init__(distribution)
        self.dims = dims
        self.domain = domain

    def __call__(self):
        t_bases = []
        for i in range(len(self.kernel.bases)):
            t_bases.append(self.kernel.bases[i].integrate(self.kernel.bases[i], self.dims, self.domain))
        self.set_kernel_bases(t_bases)

        t_GP = hgp.HilbertGP(self.kernel, solver=self._distribution.solver, transform=True)
        t_GP.domain = self._distribution.domain
        t_GP.set_kernel_parameters(self._distribution.kernel.parameters)
        t_GP.t_dim = self._distribution.dim - len(self.dims)
        self.append_ids(t_GP)

        return t_GP


class Join(Transform):

    def __init__(self, *distributions):
        super().__init__(list(*distributions))

    def __call__(self):
        raise NotImplementedError()


class Affine(Linear):

    def __init__(self, distribution, linear, translation):
        super().__init__(distribution)
        self.linear = linear
        self.translation = translation  # TODO: add translation to mean function

        self._func_valued_transform = False
        if callable(linear) or callable(translation):
            self._func_valued_transform = True

    def __call__(self):
        if self._func_valued_transform is True:
            t_bases = []
            for i in range(len(self.kernel.bases)):
                t_bases.append(self.kernel.bases[i].affine(self.kernel.bases[i], self.linear))
        else:
            t_bases = []
            for i in range(len(self.kernel.bases)):
                t_bases.append(self.kernel.bases[i].simple_affine(self.kernel.bases[i], self.linear))

        self.set_kernel_bases(t_bases)

        t_GP = hgp.HilbertGP(self.kernel, solver=self._distribution.solver, transform=True)
        t_GP.domain = self._distribution.domain
        t_GP.set_kernel_parameters(self._distribution.kernel.parameters)  # might want to check if linear has parameters
        self.append_ids(t_GP)

        return t_GP


class Sum(Linear):

    # TODO: this currently only works if the input GPs are transformations of each other. Needs to be more general where
    # the GPs aren't necessarily related...

    def __init__(self, *distributions):
        self._distribution = distributions
        self.kernel = deepcopy(self._distribution[0].kernel)
        self.kernels = [deepcopy(i.kernel) for i in self._distribution]
        self.bases = [deepcopy(i.kernel).bases for i in self._distribution]

    def __call__(self):
        @lru_cache(32)
        def t_kernel_basis_eval(x, m):
            out = self.kernels[0].basis_eval(x, m)
            for j in range(len(self.bases) - 1):
                out = out + self.kernels[j + 1].basis_eval(x, m)
            return out

        self.kernel.basis_eval = t_kernel_basis_eval
        #self.kernel.bases = self.bases

        t_GP = hgp.HilbertGP(self.kernel, solver=self._distribution[0].solver, transform=True)
        t_GP.domain = self._distribution[0].domain
        t_GP.set_kernel_parameters(self._distribution[0].kernel.parameters)

        self.append_ids(t_GP)
        return t_GP

    def append_ids(self, dist):
        for i in range(len(hgp.HilbertGP.t_ids)):
            if id(self._distribution) in hgp.HilbertGP.t_ids[i]:
                hgp.HilbertGP.t_ids[i].append(id(dist))


class InputWarp(Linear):

    def __init__(self, distribution, func):
        super().__init__(distribution)
        self.kernel_copy = deepcopy(self.kernel)
        self.func = func

    def __call__(self):
        # t_bases = []
        # for i in range(len(self.kernel.bases)):
        #     t_bases.append(self.kernel.bases[i].input_warp(self.kernel.bases[i], self.objective))
        # self.set_kernel_bases(t_bases)
        #
        # t_GP = hgp.HilbertGP(self.kernel, solver=self._distribution.solver, transform=True)
        # t_GP.domain = self._distribution.domain
        # t_GP.set_kernel_parameters(self._distribution.kernel.parameters)
        # self.append_ids(t_GP)
        # return t_GP
        # @lru_cache(32)
        def t_kernel_basis_eval(x, m):
            return self.kernel_copy.basis_eval(self.func(x), m)

        self.kernel.basis_eval = t_kernel_basis_eval

        for i in self.kernel.bases:
            i.differentiable = False
            i.integrable = False

        t_GP = hgp.HilbertGP(self.kernel, solver=self._distribution.solver, transform=True)
        t_GP.domain = self._distribution.domain
        t_GP.set_kernel_parameters(self._distribution.kernel.parameters)

        self.append_ids(t_GP)
        return t_GP

class InputWarp2(Linear):

    def __init__(self, distribution, func):
        super().__init__(distribution)
        self.kernel_copy = deepcopy(self.kernel)
        self.func = func

    def __call__(self):
        t_bases = []
        for i in range(len(self.kernel.bases)):
            t_bases.append(self.kernel.bases[i].input_warp(self.kernel.bases[i], self.func))
        self.set_kernel_bases(t_bases)

        t_GP = hgp.HilbertGP(self.kernel, solver=self._distribution.solver, transform=True)
        t_GP.domain = self._distribution.domain
        t_GP.set_kernel_parameters(self._distribution.kernel.parameters)
        self.append_ids(t_GP)
        return t_GP


class LineIntegral(Linear):

    # TODO: this transformation probably doesn't work for kernels with multiple bases (composite kernels)

    def __init__(self, distribution, r, method, add_id=True, d_r=None):
        super().__init__(distribution)
        self.r = r
        self.method = method
        self.add_id = add_id
        self.d_r = d_r

    def __call__(self):
        t_bases = []
        for i in range(len(self.kernel.bases)):
            t_bases.append(self.kernel.bases[i].line_integral(self.kernel.bases[i], self.r, self.method, self.d_r))
        self.set_kernel_bases(t_bases)

        t_GP = hgp.HilbertGP(self.kernel, solver=self._distribution.solver, transform=True, add_id=self.add_id)
        t_GP.domain = self._distribution.domain
        t_GP.set_kernel_parameters(self._distribution.kernel.parameters)
        t_GP.t_dim = 0
        self.append_ids(t_GP)
        return t_GP
