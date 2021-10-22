import torch
from copy import deepcopy
from itertools import chain, combinations

from gaussed.kernel.base import CompositeKernel


class PartialAnovaKernel(CompositeKernel):

    def __init__(self, kernel_list, interacting_dims):
        super().__init__(*kernel_list)
        self.dim = max([j for i in interacting_dims for j in i]) + 1

        self.interacting_dims = interacting_dims  # a list of lists

        for i in range(len(self.kernels)):
            self.kernels[i].dim = len(interacting_dims[i])
            self.kernels[i].bases[0].dim = len(interacting_dims[i])
            self.kernels[i].bases[0].set_dims(interacting_dims[i])

            if self.kernels[i].bases[0].dim == self.dim:
                self.kernels[i].bases[0].set_full_dims(True)
            else:
                self.kernels[i].bases[0].set_full_dims(False)

    def eval(self, x1, x2):
        raise NotImplementedError

    def basis_eval(self, x, m_list):

        basis_mat_list = []

        for i in range(len(self.kernels)):
            basis_mat_list.append(self.kernels[i].basis_eval(x, m_list[i]))

        return torch.cat(basis_mat_list, dim=1)

class AnovaKernel(PartialAnovaKernel):

    def __init__(self, kernel, dim):
        interacting_dims = self.get_full_interacting_dim_list(dim)
        kernel_list = [deepcopy(kernel) for i in range(len(interacting_dims))]
        for i in range(len(kernel_list)):
            kernel_list[i].set_dim(len(interacting_dims[i]))
        super().__init__(kernel_list, interacting_dims)

    @staticmethod
    def get_full_interacting_dim_list(dim):
        s = [i for i in range(dim)]
        iter_object = chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))
        return [list(i) for i in list(iter_object)]