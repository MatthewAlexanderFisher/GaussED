import torch
from torch.autograd import Function

"""
The following code is adapted from https://github.com/WeiWangTrento/Robust-Differentiable-SVD which is the corresponding
 repository for the paper "Robust Differentiable SVD" (https://arxiv.org/pdf/2104.03821.pdf). 
"""


def geometric_approximation(s):
    dtype = s.dtype
    I = torch.eye(s.shape[0], device=s.device).type(dtype)
    p = s.unsqueeze(-1) / s.unsqueeze(-2) - I
    p = torch.where(p < 1., p, 1. / p)
    a1 = s.repeat(s.shape[0], 1).t()
    a1_t = a1.t()
    a1 = 1. / torch.where(a1 >= a1_t, a1, - a1_t)
    a1 *= torch.ones(s.shape[0], s.shape[0], device=s.device).type(dtype) - I
    p_app = torch.ones_like(p)
    p_hat = torch.ones_like(p)
    for i in range(9):
        p_hat = p_hat * p
        p_app += p_hat
    a1 = a1 * p_app
    return a1


def clip(s):
    a1 = s.repeat(s.shape[0], 1).t()
    a1_t = a1.t()
    diff = a1 - a1_t
    diff = diff.clamp(min=0)
    diff = torch.where(diff > 0, diff.clamp(min=0.01), diff)
    diff[diff > 0] = 1. / diff[diff > 0]
    diff = diff - diff.t()
    return diff


class SafeSVD(Function):
    @staticmethod
    def forward(ctx, M):
        # s, u = th.symeig(M, eigenvectors=True, upper=True)  # s in a ascending sequence.
        ut, s, u = torch.svd(M)  # s in a descending sequence.
        # print('0-eigenvalue # {}'.format((s <= 1e-5).sum()))
        # s = torch.clamp(s, min=1e-10)  # 1e-5 - non-zero singular values...
        ctx.save_for_backward(M, u, torch.clamp(s, min=1e-16))
        return u, s

    @staticmethod
    def backward(ctx, dL_du, dL_ds):
        M, u, s = ctx.saved_tensors
        # I = th.eye(s.shape[0])
        # K_t = 1.0 / (s[None] - s[..., None] + I) - I
        K_t = geometric_approximation(s).t()
        u_t = u.t()
        dL_dM = u.mm(K_t * u_t.mm(dL_du) + torch.diag(dL_ds)).mm(u_t)
        return dL_dM


class SafeSVDClip(Function):
    @staticmethod
    def forward(ctx, M):
        # s, u = th.symeig(M, eigenvectors=True, upper=True)  # s in a ascending sequence.
        ut, s, u = torch.svd(M)  # s in a descending sequence.
        # print('0-eigenvalue # {}'.format((s <= 1e-5).sum()))
        s = torch.clamp(s, min=1e-5)
        ctx.save_for_backward(M, u, s)
        return u, s

    @staticmethod
    def backward(ctx, dL_du, dL_ds):
        M, u, s = ctx.saved_tensors
        # I = th.eye(s.shape[0])
        # K_t = 1.0 / (s[None] - s[..., None] + I) - I
        K_t = clip(s).t()
        u_t = u.t()
        dL_dM = u.mm(K_t * u_t.mm(dL_du) + torch.diag(dL_ds)).mm(u_t)
        return dL_dM


svd_safe = SafeSVD.apply
svd_clip = SafeSVDClip.apply

"""
The following code is adapted from https://github.com/pytorch/pytorch/issues/28293 which is an implementation of 
differentiating SVD with repeated singular values from the paper "Training Deep Networks with Structured Layers by
Matrix Backpropagation" (https://arxiv.org/pdf/1509.07838.pdf). 
"""


def compute_grad_V(U, S, V, grad_V):
    N = S.shape[0]
    K = svd_grad_K(S)
    S = torch.eye(N).cuda(S.get_device()) * S.reshape((N, 1))
    inner = K.T * (V.T @ grad_V)
    inner = (inner + inner.T) / 2.0
    return 2 * U @ S @ inner @ V.T


def svd_grad_K(S):
    N = S.shape[0]
    s1 = S.view((1, N))
    s2 = S.view((N, 1))
    diff = s2 - s1
    plus = s2 + s1

    # TODO Look into it
    eps = torch.ones((N, N)) * 10**(-6)
    eps = eps.cuda(S.get_device())
    max_diff = torch.max(torch.abs(diff), eps)
    sign_diff = torch.sign(diff)

    K_neg = sign_diff * max_diff

    # gaurd the matrix inversion
    K_neg[torch.arange(N), torch.arange(N)] = 10 ** (-6)
    K_neg = 1 / K_neg
    K_pos = 1 / plus

    ones = torch.ones((N, N)).cuda(S.get_device())
    rm_diag = ones - torch.eye(N).cuda(S.get_device())
    K = K_neg * K_pos * rm_diag
    return K

class CustomSVD(Function):
    """
    Custom SVD to deal with the situations when the
    singular values are equal. In this case, if dealt
    normally the gradient w.r.t to the input goes to inf.
    To deal with this situation, we replace the entries of
    a K matrix from eq: 13 in https://arxiv.org/pdf/1509.07838.pdf
    to high value.
    Note: only applicable for the tall and square matrix and doesn't
    give correct gradients for fat matrix. Maybe transpose of the
    original matrix is requires to deal with this situation. Left for
    future work.
    """
    @staticmethod
    def forward(ctx, input):
        # Note: input is matrix of size m x n with m >= n.
        # Note: if above assumption is violated, the gradients
        # will be wrong.

        U, S, V = torch.svd(input)

        ctx.save_for_backward(U, S, V)
        return U, S, V

    @staticmethod
    def backward(ctx, grad_U, grad_S, grad_V):
        U, S, V = ctx.saved_tensors
        grad_input = compute_grad_V(U, S, V, grad_V)
        return grad_input

customsvd = CustomSVD.apply
