import torch
from copy import deepcopy

from GaussED.transforms.base import Transform
import GaussED.distribution.hilbertGP as hgp
# TODO: None of these transformations will work with multioutput

class Linear(Transform):

    def __init__(self, distribution):
        super().__init__(distribution)
        self.linear = True

        self.basis = deepcopy(self._distribution.kernel.bases[0])
        self.kernel = deepcopy(self._distribution.kernel)

    def forward(self, inputs):
        raise NotImplementedError

    def append_ids(self, dist):
        for i in range(len(hgp.HilbertGP.t_ids)):
            if id(self._distribution) in hgp.HilbertGP.t_ids[i]:
                hgp.HilbertGP.t_ids[i].append(id(dist))
                return

    def set_kernel_basis(self, new_basis):
        self.kernel.bases = [new_basis]


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

    def __init__(self, distribution, dims, orders):
        super().__init__(distribution)
        self.dims = dims
        self.orders = orders

    def __call__(self):
        diff_orders = self.basis.diff_orders

        for j in range(len(self.dims)):
            # if we want dims to start with, replace j with j-1 in the following
            diff_orders[self.dims[j]] = diff_orders[self.dims[j]] + self.orders[j]

        t_basis = self.basis.differentiate(self.basis, [0], [2])
        self.set_kernel_basis(t_basis)

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
        t_basis = self.basis.integrate(self.basis, self.dims, self.domain)
        self.set_kernel_basis(t_basis)

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
            t_basis = self.basis.affine(self.basis, self.linear)
        else:
            t_basis = self.basis.simple_affine(self.basis, self.linear)

        self.set_kernel_basis(t_basis)

        t_GP = hgp.HilbertGP(self.kernel, solver=self._distribution.solver, transform=True)
        t_GP.domain = self._distribution.domain
        t_GP.set_kernel_parameters(self._distribution.kernel.parameters)  # might want to check if linear has parameters
        self.append_ids(t_GP)

        return t_GP


class Project(Linear):

    def __init__(self, distribution, dims):
        super().__init__(distribution)
        self.dims = dims

    def __call__(self):
        t_basis = self.basis.project(self.dims)  # TODO: make this work and static method this
        self.set_kernel_basis(t_basis)

        t_GP = hgp.HilbertGP(self.kernel, solver=self._distribution.solver, transform=True)
        t_GP.domain = self._distribution.domain
        t_GP.set_kernel_parameters(self._distribution.kernel.parameters)
        self.append_ids(t_GP)
        return t_GP


class Sum(Linear):

    def __init__(self, *distributions):
        self._distribution = distributions
        self.bases = [i.kernel.basis for i in self._distribution]

        self.basis = self._distribution[0].kernel.basis
        self.kernel = deepcopy(self._distribution[0].kernel)

    def __call__(self):
        t_basis = self.basis.sum(*self.bases)
        self.set_kernel_basis(t_basis)

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
        self.func = func

    def __call__(self):
        t_basis = self.basis.input_warp(self.basis, self.func)

        self.set_kernel_basis(t_basis)

        t_GP = hgp.HilbertGP(self.kernel, solver=self._distribution.solver, transform=True)
        t_GP.domain = self._distribution.domain
        t_GP.set_kernel_parameters(self._distribution.kernel.parameters)
        self.append_ids(t_GP)
        return t_GP

class LineIntegral(Linear):

    def __init__(self, distribution, r, method, d_r=None):
        super().__init__(distribution)
        self.r = r
        self.method = method
        self.d_r = d_r

    def __call__(self):
        t_basis = self.basis.line_integral(self.basis, self.r, self.method, self.d_r)
        self.set_kernel_basis(t_basis)

        t_GP = hgp.HilbertGP(self.kernel, solver=self._distribution.solver, transform=True)
        t_GP.domain = self._distribution.domain
        t_GP.set_kernel_parameters(self._distribution.kernel.parameters)
        t_GP.t_dim = 0
        self.append_ids(t_GP)
        return t_GP
