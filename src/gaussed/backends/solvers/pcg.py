from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple, Dict
import jax.numpy as jnp
from jax import Array
import jax

from gaussed.backends.solvers.base import Solver, Factor
from gaussed.linops import LinearOp, IdentityOp, AsLinearOp

@dataclass
class PCGFactor(Factor):
    A: LinearOp
    _dtype: Array
    M_inv: Optional[LinearOp] = None   # preconditioner (approx inv)
    rtol: float = 1e-6
    atol: float = 1e-6
    maxit: int = 512

    def solve(self, u: Array) -> Array:
        # plug any CG/PCG implementation you like
        return pcg(self.A.mv,
                    u,
                    M_inv=self.M_inv,
                    rtol=self.rtol,
                    atol=self.atol,
                    maxiter=self.maxit
                    )

    def solve_blocks(self, B: Array) -> Array:
        # solve each column
        return jax.vmap(self.solve, in_axes=1, out_axes=1)(B)
    
    def logdet(self) -> Array:
        raise NotImplementedError("logdet not implemented for PCGFactor")
    
    def dtype(self) -> Array: 
        return self._dtype

@dataclass
class PCGSolver(Solver):
    precond: Optional[LinearOp] = None
    tol: float = 1e-6
    maxit: int = 512

    def factor(self, A: Array | LinearOp) -> Factor:
        op = AsLinearOp(A)
        return PCGFactor(op, op.to_dense().dtype, self.precond, self.tol, self.maxit)


def pcg(
    A: LinearOp,
    b: Array,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-6,
    wce_tol: float = 0.0,
    maxiter: int = 1_000,
    M_inv: Optional[LinearOp] = None,
    x0: Optional[Array] = None,
    return_metrics: bool = False
) -> Tuple[Array, Dict] | Array:
    r"""
    Solve the symmetric system :math:`Ax = b` using Preconditioned Conjugate Gradient (PCG),
    optionally tracking residuals.

    This function dispatches to a JIT-compiled core:

    - If both ``return_history=True`` and ``f_vals`` is supplied, uses
      a PCG core that also computes the quadrature estimates at each iteration.
    - Otherwise, uses the history-only core or the plain solver.

    Args:
        A:
            Linear operator representing :math:`A` (assumed symmetric positive semi-definite).
        b (Array):
            Right-hand side vector.
        rtol (float, optional):
            Relative convergence tolerance.
        atol (float, optional):
            Absolute convergence tolerance.
        wce_tol (float, optional):
            Tolerance for worst-case error early stopping. If WCE drops below this, iteration stops.
            Defaults to``0.0`` (disabled).
        maxiter (int, optional):
            Maximum number of iterations.
        M_inv (Array, optional):
            Preconditioner (approximate inverse of :math:`A`). Defaults to ``None`` (no preconditioning).
        x0 (Array, optional):
            Initial guess for ``x``, shape ``(n,)``. If ``None``, the zero vector is used.
        return_history (bool):
            If ``True``, return per-iteration histories for residual norm and WCE.
            If ``False``, only the final values are returned.
    Returns:
        tuple:
            A tuple whose length and contents depend on the flags:

            - **x** (``Array``):
            Final solution :math:`x_k`. Shape: ``(n,)``.
            - **k** (``int``):
            Number of iterations performed.
            - **res_hist** (:math:`(k,)`, optional):
            Residual norms :math:`\|r_i\|` at each iteration ; only if ``return_history=True``.
    """

    n = b.shape[0]

    if M_inv is None:
        M_inv = IdentityOp(n, b.dtype)

    x, metrics = _pcg_core(
        A,
        b,
        rtol=rtol,
        atol=atol,
        wce_tol=wce_tol,
        maxiter=maxiter,
        M_inv=M_inv,
        x0=x0,
    )
    if return_metrics is True:
        return x, metrics

    return x


# PCG without history JIT-compiled core function
@partial(jax.jit, static_argnames=("rtol", "atol", "wce_tol", "maxiter"))
def _pcg_core(
    A: Array,
    b: Array,
    M_inv: Array,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-6,
    wce_tol: float = 0.0,
    maxiter: int = 1000,
    x0: Optional[Array] = None,
) -> Tuple[Array, Dict] | Array:
    r"""
    Core implementation of the Preconditioned Conjugate Gradient (PCG) method
    for solving the linear system :math:`\mathbf{A}x = b`.

    This internal function is JIT-compiled for performance and supports optional
    left preconditioning using a symmetric positive definite matrix :math:`\mathbf{M}^{-1}`.
    It also tracks the worst-case error (WCE), which provides a computable
    upper bound on the error of kernel-based integration estimates.

    Returns the values of the residual norms and worst-case RKHS quadrature errors at each
    iteration.

    Args:
        A:
            Linear operator or matrix representing :math:`A`. Must be symmetric
            positive semi-definite and support `A @ x`.
        b (Array):
            Right-hand side vector.
        rtol (float, optional):
            Relative tolerance for convergence.
        atol (float, optional):
            Absolute tolerance for convergence.
        wce_tol (float, optional):
            Early stopping criterion based on the worst-case error. Iteration terminates
            when WCE falls below this threshold. Defaults to 0.0 (disabled).
        maxiter (int, optional):
            Maximum number of iterations.
        M_inv (Array, optional):
            Preconditioner matrix :math:`\mathbf{M}^{-1}`. Must be symmetric positive definite.
        x0 (Array, optional):
            Initial guess for the solution. If not provided, a zero vector is used.

    Returns:
        tuple:
            A 4-tuple containing:

            - **x** (Array): Final solution :math:`x_k` to the system.
            - **k** (int): Number of iterations performed.
            - **residuals** (Array): Norms of the residual at termination :math:`\|r_k\|`. Shape: (1,).
            - **wce** (Array): Worst-case error estimate at termination :math:`\sigma(w_k)`. Shape: (1,).
    """

    x = jnp.zeros_like(b) if x0 is None else x0
    r = b - A @ x
    z = M_inv @ r
    rho = r @ z
    res0 = jnp.linalg.norm(r)
    eps = jnp.finfo(b.dtype).tiny

    # state = (k, x, r, z, rho, res, w, done)
    state = (0, x, r, z, rho, res0, False)

    def cond(state):
        k, _, r, _, _, res, w, _ = state
        return ~((res <= atol) | (res <= rtol * res0) | (w <= wce_tol) | (k >= maxiter))

    def body(state):
        k, x, r, z, rho, res, _, _ = state

        Az = A @ z
        alpha = rho / (z @ Az + eps)
        x = x + alpha * z
        r = r - alpha * Az
        rho_new = r @ (M_inv @ r)
        beta = rho_new / (rho + eps)
        z = M_inv @ r + beta * z
        res = jnp.linalg.norm(r)

        return (k + 1, x, r, z, rho_new, res, False)

    k_final, x_final, r_final, z_final, rho_final, res_final, _ = (
        jax.lax.while_loop(cond, body, state)
    )

    return x_final

