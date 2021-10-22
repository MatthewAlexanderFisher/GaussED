import torch


def nearestSPD(mat):
    """ Calculates nearest semi-positive definite matrix w.r.t. the Frobenius norm (algorithm based on Nick Higham's
    "Computing the nearest correlation matrix - a problem from finance".

    Args:
        mat ([torch.Tensor]): [Input matrix.]

    Returns:
        [torch.Tensor]: [The symmetric positive definite matrix that is closest to input in Frobenius norm.]
    """

    B = (mat + mat.T) / 2
    _, s, V = torch.svd(B)
    H = torch.matmul(V, V.mul(s).T)

    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    I = torch.eye(mat.shape[0])
    k = 1
    while not isPD(A3):
        mineig = torch.min(torch.symeig(A3)[0].T[0])
        A3 += I * (-mineig * k ** 2)
        k += 1
    if torch.norm(mat - A3) / torch.norm(A3) > 10:
        print(
            "Matrix failed to be positive definite, distance in Frobenius norm: ",
            torch.norm(mat - A3, ord="fro") / torch.norm(A3, ord="fro"),
        )
    return A3


def isPD(B):
    """Check whether a matrix is positive definite.

    Args:
        B ([torch.Tensor]): [Input matrix.]

    Returns:
        [bool]: [Returns True if matrix is positive definite, otherwise False.]
    """
    try:
        _ = torch.cholesky(B)
        return True
    except RuntimeError:
        return False
