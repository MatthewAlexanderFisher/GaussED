from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Tuple, Optional

import jax
import jax.numpy as jnp
from jax import Array

from gaussed.linops import LinearOp, AsLinearOp
from gaussed.backends.solvers.linear_solver import LinearSolverState, SqrtFn, LogdetFn

# ---------------------------
# Lanczos tridiagonalisation
# ---------------------------

def _lanczos_tridiag(
    mv: Callable[[Array], Array],
    v0: Array,
    k: int,
) -> Tuple[Array, Array]:
    """
    Run k steps of (symmetric) Lanczos with starting vector v0 (not necessarily unit).
    Returns:
        alphas: (k,)  main diagonal of T_k
        betas:  (k-1,) sub/super diagonal of T_k
    Notes:
        - Three-term recurrence (no explicit reorthogonalisation).
        - For SPD A this is typically robust; increase jitter on A if needed.
        - Fixed iteration count -> JIT-friendly (no dynamic shapes).
    """
    v0 = jnp.asarray(v0)

    v0_norm = jnp.linalg.norm(v0)
    is_zero = v0_norm == 0

    # q0 := v0 / ||v0||  (or zeros if v0==0)
    q0 = jax.lax.cond(
        is_zero,
        lambda _: jnp.zeros_like(v0),
        lambda _: v0 / v0_norm,
        operand=None,
    )

    q_im1 = jnp.zeros_like(q0)                       # q_{-1}
    beta_im1 = jnp.zeros((), dtype=v0.dtype)         # scalar with right dtype

    alphas = jnp.zeros((k,), dtype=v0.dtype)
    betas  = jnp.zeros((k-1,), dtype=v0.dtype)

    def body(i, carry):
        q_im1, q_i, beta_im1, alphas, betas = carry
        # w = A q_i - beta_{i-1} q_{i-1}
        w = mv(q_i) - beta_im1 * q_im1
        alpha_i = jnp.vdot(q_i, w).real  # scalar; A symmetric
        w = w - alpha_i * q_i
        beta_i = jnp.linalg.norm(w)

        alphas = alphas.at[i].set(alpha_i)
        betas  = jax.lax.cond(i < k-1,
                              lambda b: b.at[i].set(beta_i),
                              lambda b: b, betas)

        # Next q
        q_ip1 = jnp.where(beta_i > 0, w / beta_i, jnp.zeros_like(w))
        return (q_i, q_ip1, beta_i, alphas, betas)

    q_im1, q_i, beta_im1, alphas, betas = jax.lax.fori_loop(
        0, k, body, (q_im1, q0, beta_im1, alphas, betas)
    )
    return alphas, betas

def _tridiag_from_ab(alphas: Array, betas: Array) -> Array:
    """Build symmetric tridiagonal T from diagonals."""
    k = alphas.shape[0]
    T = jnp.diag(alphas)
    T = T + jnp.diag(betas, 1)
    T = T + jnp.diag(betas, -1)
    return T


def lanczos_apply_f(
    mv: Callable[[Array], Array],
    v: Array,
    k: int,
    f: Callable[[Array], Array],
) -> Array:
    """
    Approximate y ≈ f(A) v via Lanczos:
        1) Build T_k from v (k steps).
        2) Compute c = f(T_k) e1 in the Krylov basis.
        3) Re-run Lanczos to accumulate y = ||v|| * sum_i c_i * q_i.

    Memory: O(n) (doesn't store the whole V_k basis).
    Work:   k matvecs + O(k^3) small dense eigendecomp; second pass k matvecs.
    """
    vnorm = jnp.linalg.norm(v)
    # If v is zero, return zero
    def _zero(_):
        return jnp.zeros_like(v)
    def _nonzero(_):
        # Pass 1: coefficients from small T_k
        alphas, betas = _lanczos_tridiag(mv, v, k)
        T = _tridiag_from_ab(alphas, betas)
        evals, Q = jnp.linalg.eigh(T)                         # (k,), (k,k)
        e1 = jnp.zeros((k,), dtype=v.dtype).at[0].set(1.0)
        # c = f(T) e1 = Q f(Λ) Q^T e1
        u = Q.T @ e1
        c = Q @ (f(evals) * u)                                # (k,)

        # Pass 2: reconstruct y = ||v|| * V_k c
        q0 = v / vnorm
        q_im1 = jnp.zeros_like(q0)
        beta_im1 = jnp.array(0.0, dtype=v.dtype)
        y = jnp.zeros_like(v)

        def body(i, carry):
            q_im1, q_i, beta_im1, y = carry
            y = y + c[i] * q_i
            # Lanczos step
            w = mv(q_i) - beta_im1 * q_im1
            alpha_i = jnp.vdot(q_i, w).real
            w = w - alpha_i * q_i
            beta_i = jnp.linalg.norm(w)
            q_ip1 = jnp.where(beta_i > 0, w / beta_i, jnp.zeros_like(w))
            return (q_i, q_ip1, beta_i, y)

        q_im1, q_i, beta_im1, y = jax.lax.fori_loop(0, k, body, (q_im1, q0, beta_im1, y))
        return vnorm * y

    return jax.lax.cond(vnorm == 0, _zero, _nonzero, operand=None)


# log-determinant via Hutchinson + Lanczos

def _slq_single_probe(
    mv: Callable[[Array], Array],
    n: int,
    k: int,
    key: Array,
    rademacher: bool = True,
) -> Array:
    """
    One SLQ estimate of z^T log(A) z using k-step Lanczos.
    Returns a scalar (not yet normalised by number of probes).
    """
    if rademacher:
        z = jax.random.randint(key, (n,), 0, 2, dtype=jnp.int32)
        z = (2 * z - 1).astype(jnp.result_type(0.0))  # ±1
        z_norm2 = jnp.array(n, dtype=z.dtype)         # ||z||^2 = n exactly
    else:
        z = jax.random.normal(key, (n,))
        z_norm2 = jnp.vdot(z, z).real

    # Use unit vector for Lanczos
    v0 = z / jnp.sqrt(z_norm2)
    alphas, betas = _lanczos_tridiag(mv, v0, k)
    T = _tridiag_from_ab(alphas, betas)
    evals, Q = jnp.linalg.eigh(T)
    e1 = jnp.zeros((k,), dtype=z.dtype).at[0].set(1.0)
    u = Q.T @ e1
    quad = jnp.sum((u * u) * jnp.log(jnp.clip(evals, a_min=jnp.finfo(evals.dtype).tiny)))
    return z_norm2 * quad

def slq_logdet(
    op: LinearOp,
    *,
    k: int = 32,
    n_probe: int = 16,
    key: Array,
    rademacher: bool = True,
    eps: float = 0.0,
) -> Array:
    """
    Estimate log|A + eps I| via Stochastic Lanczos Quadrature with n_probe probes.
    Works best for SPD A; add eps>0 if needed.
    """
    mv = (lambda x: op.mv(x) + eps * x) if eps != 0.0 else op.mv
    n = op.shape[0]
    keys = jax.random.split(key, n_probe)
    probe_est = jax.vmap(lambda k_: _slq_single_probe(mv, n, k, k_, rademacher))(keys)
    # Average the Hutchinson estimates to get tr(log A)
    return jnp.mean(probe_est)


# ===== LinearSolver hooks =====

def lanczos_inv_sqrt_hook(k: int = 32, eps: float = 0.0) -> SqrtFn:
    """
    SqrtFn: apply (A + eps I)^{-1/2} to rhs via two-pass Lanczos.
    Stateless: preserves incoming solver state.
    """
    def _sqrt(op: LinearOp, rhs: Array, st: LinearSolverState):
        mv = (lambda x: op.mv(x) + eps * x) if eps != 0.0 else op.mv
        if rhs.ndim == 1:
            y = lanczos_apply_f(mv, rhs, k, f=lambda x: 1.0 / jnp.sqrt(jnp.clip(x, a_min=jnp.finfo(x.dtype).tiny)))
        else:
            y = jax.vmap(lambda col: lanczos_apply_f(mv, col, k, f=lambda x: 1.0 / jnp.sqrt(jnp.clip(x, a_min=jnp.finfo(x.dtype).tiny))),
                         in_axes=1, out_axes=1)(rhs)
        return y, st  # keep caches from other hooks intact
    return _sqrt


# --- logdet hook via SLQ (stateless; key captured in closure) ---

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SLQParams:
    k: int
    n_probe: int
    key: Array
    rademacher: bool
    eps: float
    def tree_flatten(self): return (self.key,), (self.k, self.n_probe, self.rademacher, self.eps)
    @classmethod
    def tree_unflatten(cls, aux, children):
        k, n_probe, rademacher, eps = aux
        (key,) = children
        return cls(k, n_probe, key, rademacher, eps)

def slq_logdet_hook(
    *,
    k: int = 32,
    n_probe: int = 16,
    key: Array,
    rademacher: bool = True,
    eps: float = 0.0,
) -> LogdetFn:
    """
    LogdetFn: estimate log|A + eps I| via SLQ.
    Stateless w.r.t. other caches: returns the incoming state unchanged.
    If you want fresh probes each call, pass a different key when constructing
    a new hook (or extend state to carry/advance the key).
    """
    params = SLQParams(k, n_probe, key, rademacher, eps)

    def _logdet(op: LinearOp, st: LinearSolverState):
        val = slq_logdet(op, k=params.k, n_probe=params.n_probe, key=params.key,
                         rademacher=params.rademacher, eps=params.eps)
        return val, st
    return _logdet
