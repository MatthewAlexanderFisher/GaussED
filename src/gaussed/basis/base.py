from torch import nn


class Basis(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        """Forward the forward computation .

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError
