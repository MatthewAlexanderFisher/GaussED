import torch


def sum_tensor_gen(dim, m):
    """Generate the summation matrix for the Laplace basis object.

    Args:
        dim ([type]): [description]
        m ([type]): [description]

    Returns:
        [type]: [description]
    """
    return torch.stack(
        torch.meshgrid(*[torch.arange(1, m + 1, 1) for i in range(dim)])
    ).T.reshape(m ** dim, dim)
