from torch import nn


class Basis(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError
