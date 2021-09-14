import torch

from GaussED.distribution import HilbertGP

class Loss:

    def __init__(self, dim=0, mesh=None):
        self.dim = dim
        self.mesh = mesh

    def __call__(self, q1, q2):
        raise NotImplementedError


class L2(Loss):

    def __init__(self, qoi, mesh=None, use_mesh=None):
        super().__init__(qoi.t_dim, mesh)

        self.qoi = qoi
        if use_mesh is None:
            if type(qoi) is HilbertGP:
                self.use_mesh = True
            else:
                self.use_mesh = False
        else:
            self.use_mesh = use_mesh

    def __call__(self, q1, q2):
        if self.use_mesh is True:
            return torch.mean((q1(self.mesh) - q2(self.mesh)).pow(2), dim=-2)
        else:
            return (q1 - q2).pow(2)  # this might not work

    def vec_call(self, q1, q2):
        if self.use_mesh is True:
            return torch.mean((q1(self.mesh).T - q2(self.mesh)).T.pow(2), dim=-2)
        else:
            return (q1 - q2).T.pow(2)
