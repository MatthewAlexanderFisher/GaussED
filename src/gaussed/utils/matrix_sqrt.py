import torch
from torch.autograd import Function
import numpy as np
import scipy.linalg


class MatrixSquareRoot(Function):

    """ 
    A differentiable square root of a positive definite matrix. 
    NOTE: matrix square root is not differentiable for matrices with zero eigenvalues.
    """

    @staticmethod
    def forward(ctx, input):
        """Compute the square root of the input.

        Args:
            ctx ([type]): [Autograd context]
            input ([type]): [description]

        Returns:
            [torch.Tensor]: [description]
        """
        m = input.detach().cpu().numpy().astype(np.float_)
        sqrtm = torch.from_numpy(scipy.linalg.sqrtm(m).real).to(input)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            (sqrtm,) = ctx.saved_tensors
            sqrtm = sqrtm.data.cpu().numpy().astype(np.float_)
            gm = grad_output.data.cpu().numpy().astype(np.float_)

            # Given a positive semi-definite matrix X,
            # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
            # matrix square root dX^{1/2} by solving the Sylvester equation:
            # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)

            grad_input = torch.from_numpy(grad_sqrtm).to(grad_output)
        return grad_input


sqrtm = MatrixSquareRoot.apply


class MatrixSVDSquareRoot(Function):
    """
    Computes the square root of a matrix using singular value decomposition. Computes the backward gradient pass using
    scipy's solve_sylvester
    """

    @staticmethod
    def forward(ctx, input):
        U, S, V = torch.svd(input)
        sqrtm = V.mul(torch.sqrt(S))
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            (sqrtm,) = ctx.saved_tensors
            sqrtm = sqrtm.data.cpu().numpy().astype(np.float_)
            gm = grad_output.data.cpu().numpy().astype(np.float_)
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)

            grad_input = torch.from_numpy(grad_sqrtm).to(grad_output)
        return grad_input


svd_sqrtm = MatrixSVDSquareRoot.apply
