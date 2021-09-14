import torch
# import numpy as np


def sum_tensor_gen(dim, m):
    return torch.stack(torch.meshgrid(*[torch.arange(1, m + 1, 1) for i in range(dim)])).T.reshape(m ** dim, dim)
