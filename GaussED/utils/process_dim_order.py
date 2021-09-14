import torch


def process_dim_order(dim_order, dim):
    """
    Helper function for differentiation and integral_transform of HilbertGP
    :param dim_order: differentiation transformation input
    :param dim: dimension
    :return: permutation tensor, splitting list, grouped orders, objective orders
    """
    dim_order = torch.LongTensor(dim_order)
    if dim_order.T[0][-1] > dim:
        raise RuntimeError("Dimension of order exceeds dimension of input")

    order_tensor = torch.zeros(dim, dtype=torch.long)
    order_tensor[dim_order.T[0] - 1] = dim_order.T[1]

    sorted_order, perm_order = torch.sort(order_tensor)

    permutation = torch.sort(perm_order)[1]
    split_tensor = torch.bincount(order_tensor)  # torch.bincount counts the number of unique positive integer values
    split_list = list(split_tensor[split_tensor != 0])

    grouped_orders = torch.unique(sorted_order)
    func_order = torch.remainder(grouped_orders, 4)  # used to determine which function is used in the derivative

    return permutation, split_list, grouped_orders, func_order
