import torch

from GaussED.kernel.base import CompositeKernel


class AnovaKernel(CompositeKernel):

    def __init__(self, kernel_list, interacting_dims):
        super().__init__(*kernel_list)
        self.dim = max([j for i in interacting_dims for j in i]) + 1

        self.interacting_dims = interacting_dims  # a list of lists

        for i in range(len(self.kernels)):
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

