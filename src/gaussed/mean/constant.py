import torch

from gaussed.mean.base import Mean


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
        """Evaluate the mean function.

        Args:
            x ([torch.Tensor]): Input Tensor

        Returns:
            [torch.Tensor]: Output Tensor
        """
        return torch.zeros(x.shape)
