from dataclasses import dataclass
from typing import Tuple, Optional
import jax
import jax.numpy as jnp
from jax import Array

from gaussed.backends.solvers.linear_solver import LinearSolverState, SolveFn, SqrtFn, LogdetFn
from gaussed.linops import LinearOp, AsLinearOp
from gaussed.types import LinearLike

# ===== SVD Cache =====

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SVDCache:
    U: Array   # (n, r)
    s: Array   # (r,)
    V: Array   # (m, r)

    def tree_flatten(self): return (self.U, self.s, self.V), ()
    @classmethod
    def tree_unflatten(cls, aux, ch): U,s,V = ch; return cls(U,s,V)

def svd_factor_from_op(
    op: LinearOp,
    jitter: float = 0.0,
    symmetrise: bool = True,
) -> Tuple[Array, Array, Array]:
    """
    Compute (optionally regularised) SVD of the operator's dense matrix.

    If `symmetrise=True`, uses K := 0.5 (K + K^T), which is standard when
    you intend to treat A as SPD/PSD. `jitter>0` applies Tikhonov regularisation
    (A <- A + jitter * I) before the SVD for stability.

    Returns
    -------
    U : (n, r), s : (r,), V : (m, r)
        Thin SVD so that A ≈ U @ diag(s) @ V.T
        For square symmetric A, U ≈ V and s are eigenvalues (nonnegative).
    """
    K = op.to_dense()
    if symmetrise:
        K = 0.5 * (K + K.T)
    if jitter:
        K = K + jitter * jnp.eye(K.shape[0], dtype=K.dtype)
    U, s, Vt = jnp.linalg.svd(K, full_matrices=False)
    V = Vt.T
    return U, s, V

# ===== Hooks: SVD-based solve / sqrt / logdet =====

def svd_solve_hook(
    jitter: float = 0.0,
    symmetrise: bool = True,
    rcond: float = 0.0,  # relative cutoff for tiny singular values
) -> SolveFn:
    """
    Solve A x ≈ b using the (regularised) pseudoinverse from SVD:
      x = V diag(1/s_clipped) U^T b
    If `symmetrise=True`, treat A as symmetric and pre-symmetrise A.
    """
    def _solve(op: LinearOp, rhs: Array, st: LinearSolverState):
        cache = st.cache
        if isinstance(cache, SVDCache):
            U, s, V = cache.U, cache.s, cache.V
        else:
            U, s, V = svd_factor_from_op(op, jitter=jitter, symmetrise=symmetrise)
            cache = SVDCache(U, s, V)

        # Build reciprocal with cutoff (avoids blow-ups on near-zero s)
        smax = jnp.max(s) if s.size else jnp.array(1.0, rhs.dtype)
        cutoff = rcond * smax
        s_inv = jnp.where(s > cutoff, 1.0 / s, 0.0)

        # Handle vector (n,) or matrix (n,k) RHS uniformly:
        x = V @ (s_inv[:, None] * (U.T @ rhs))
        return x, LinearSolverState(cache)
    return _solve

def svd_sqrt_hook(
    jitter: float = 0.0,
    symmetrise: bool = True,
    rcond: float = 0.0,
) -> SqrtFn:
    """
    Apply an inverse square-root based on SVD:
      y = A^{-1/2} b ≈ V diag(1/s^{1/2}) U^T b
    Correct as A^{-1/2} when A is symmetric positive definite (then U=V, s=eigs).
    With `symmetrise=True`, we enforce symmetry before factoring.
    """
    def _sqrt(op: LinearOp, rhs: Array, st: LinearSolverState):
        cache = st.cache
        if isinstance(cache, SVDCache):
            U, s, V = cache.U, cache.s, cache.V
        else:
            U, s, V = svd_factor_from_op(op, jitter=jitter, symmetrise=symmetrise)
            cache = SVDCache(U, s, V)

        smax = jnp.max(s) if s.size else jnp.array(1.0, rhs.dtype)
        cutoff = rcond * smax
        s_isqrt = jnp.where(s > cutoff, 1.0 / jnp.sqrt(s), 0.0)

        y = V @ (s_isqrt[:, None] * (U.T @ rhs))
        return y, LinearSolverState(cache)
    return _sqrt

def svd_logdet_hook(
    jitter: float = 0.0,
    symmetrise: bool = True,
    pseudo: bool = False,   # if True, compute pseudo-logdet (sum log of positive s)
    rcond: float = 0.0,
) -> LogdetFn:
    """
    Compute log|A| from SVD. For SPD matrices (with or without jitter), logdet(A) = sum log s.
    If `pseudo=True`, returns sum over log(s_i) for s_i > rcond * max(s), useful for PSD.
    Otherwise, if any s <= 0 (non-SPD), returns NaN to avoid misleading values.
    """
    def _logdet(op: LinearOp, st: LinearSolverState):
        cache = st.cache
        if isinstance(cache, SVDCache):
            U, s, V = cache.U, cache.s, cache.V
        else:
            U, s, V = svd_factor_from_op(op, jitter=jitter, symmetrise=symmetrise)
            cache = SVDCache(U, s, V)

        smax = jnp.max(s) if s.size else jnp.array(1.0, s.dtype)
        if pseudo:
            mask = s > (rcond * smax)
            val = jnp.sum(jnp.log(jnp.where(mask, s, 1.0)))  # ignore tiny/zero singular values
        else:
            # Strict SPD requirement
            if jnp.any(s <= 0):
                val = jnp.array(jnp.nan, dtype=s.dtype)
            else:
                val = jnp.sum(jnp.log(s))
        return val, LinearSolverState(cache)
    return _logdet

# ===== Block-append update SVD (Brand algorithm) =====

def svd_extend_cols(cache: SVDCache, B: Array):
    """Incremental SVD for [A, B], appending k columns B (n,k)."""
    U, s, V = cache.U, cache.s, cache.V          # (n,r), (r,), (m,r)
    # Project/orthogonalise new columns
    P = U.T @ B                                  # (r,k)
    R = B - U @ P                                # (n,k)
    Qr, Rr = jnp.linalg.qr(R, mode="reduced")    # Qr:(n,q), Rr:(q,k)
    q = Rr.shape[0]
    r = s.shape[0]
    # Build small core K = [[Σ, P],[0, Rr]]
    Sigma = jnp.diag(s)                          # (r,r)
    top = jnp.concatenate([Sigma, P], axis=1)    # (r, r+k)
    bot = jnp.concatenate([jnp.zeros((q, r), Rr.dtype), Rr], axis=1)  # (q, r+k)
    K = jnp.concatenate([top, bot], axis=0)      # (r+q, r+k)
    # SVD of small core
    Uk, sk, VkT = jnp.linalg.svd(K, full_matrices=False)  # Uk:(r+q, r'), VkT:(r', r+k)
    Vk = VkT.T
    # Expand U and V bases
    U_aug = jnp.concatenate([U, Qr], axis=1)     # (n, r+q)
    V_pad_top = jnp.concatenate([V, jnp.zeros((V.shape[0], Vk.shape[0]-V.shape[1]), V.dtype)], axis=1)  # (m, r+q) if needed
    V_pad_bot = jnp.concatenate([jnp.zeros((B.shape[1], V.shape[1]), V.dtype),
                                 jnp.eye(B.shape[1], dtype=V.dtype)], axis=1)                             # (k, r+q)
    V_aug = jnp.concatenate([V_pad_top, V_pad_bot], axis=0)  # (m+k, r+q)
    # Update factors
    U_new = U_aug @ Uk                           # (n, r')
    V_new = V_aug @ Vk                           # (m+k, r')
    return SVDCache(U_new, sk, V_new)
