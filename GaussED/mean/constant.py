import torch

from GaussED.mean.base import Mean


class ConstantMean(Mean):

    def __init__(self, c):
        super().__init__()
        self.c = c
        self.parameters = [c]

    def eval(self, x):
        return torch.zeros(x.shape) + self.c


class ZeroMean(ConstantMean):

    def __init__(self):
        super().__init__(c=0)
        self.parameters = []

    def eval(self, x):
        return torch.zeros(x.shape)
