import torch


def process_dim_order(dim_order, dim):
    """Helper function for computing the derivatives of the basis functions in the Laplace basis object

    Args:
        dim_order ([type]): [description]
        dim ([type]): [description]

    Raises:
        RuntimeError: [description]

    Returns:
        [type]: [description]
    """
    dim_order = torch.LongTensor(dim_order)
    if dim_order.T[0][-1] > dim:
        raise RuntimeError("Dimension of order exceeds dimension of input")

    order_tensor = torch.zeros(dim, dtype=torch.long)
    order_tensor[dim_order.T[0] - 1] = dim_order.T[1]

    sorted_order, perm_order = torch.sort(order_tensor)

    permutation = torch.sort(perm_order)[1]
    split_tensor = torch.bincount(
        order_tensor
    )  # torch.bincount counts the number of unique positive integer values
    split_list = list(split_tensor[split_tensor != 0])

    grouped_orders = torch.unique(sorted_order)
    func_order = torch.remainder(
        grouped_orders, 4
    )  # used to determine which function is used in the derivative

    return permutation, split_list, grouped_orders, func_order
