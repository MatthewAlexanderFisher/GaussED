from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jax import Array

from gaussed.backends.solvers.linear_solver import LinearSolverState, SolveFn, SqrtFn, LogdetFn
from gaussed.linops import LinearOp, AsLinearOp
from gaussed.types import LinearLike

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CholCache:
    L: Array  # lower-triangular factor

    def tree_flatten(self):
        return (self.L,), ()
    @classmethod
    def tree_unflatten(cls, aux, children):
        (L,) = children
        return cls(L)

def cholesky_factor_from_op(op: LinearOp, jitter: float = 0.0, symmetrise: bool = True) -> Array:
    K = op.to_dense()
    if symmetrise:
        K = 0.5 * (K + K.T)
    if jitter:
        K = K + jitter * jnp.eye(K.shape[0], dtype=K.dtype)
    return jnp.linalg.cholesky(K)

# Hook: SOLVE
def chol_solve_hook(jitter: float = 0.0, symmetrise: bool = True) -> SolveFn:
    def _solve(op: LinearOp, rhs: Array, st: LinearSolverState):
        cache = st.cache
        if isinstance(cache, CholCache):
            L = cache.L
        else:
            L = cholesky_factor_from_op(op, jitter, symmetrise)
            cache = CholCache(L)
        y = jax.scipy.linalg.solve_triangular(L, rhs, lower=True, trans='N')
        x = jax.scipy.linalg.solve_triangular(L, y, lower=True, trans='T')
        return x, LinearSolverState(cache)
    return _solve

# Hook: SQRT (apply A^{-1/2} = L^{-T})
def chol_sqrt_hook(jitter: float = 0.0, symmetrise: bool = True) -> SqrtFn:
    def _sqrt(op: LinearOp, rhs: Array, st: LinearSolverState):
        cache = st.cache
        if isinstance(cache, CholCache):
            L = cache.L
        else:
            L = cholesky_factor_from_op(op, jitter, symmetrise)
            cache = CholCache(L)
        x = jax.scipy.linalg.solve_triangular(L, rhs, lower=True, trans='T')
        return x, LinearSolverState(cache)
    return _sqrt

# Hook: LOGDET
def chol_logdet_hook(jitter: float = 0.0, symmetrise: bool = True) -> LogdetFn:
    def _logdet(op: LinearOp, st: LinearSolverState):
        cache = st.cache
        if isinstance(cache, CholCache):
            L = cache.L
        else:
            L = cholesky_factor_from_op(op, jitter, symmetrise)
            cache = CholCache(L)
        val = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
        return val, LinearSolverState(cache)
    return _logdet

# block-append update for SED
def chol_extend_block(cache: CholCache, B: Array, C: Array, jitter: float = 0.0) -> CholCache:
    """
    Incrementally extend a Cholesky factorisation to include additional rows/columns.

    Suppose we already have the Cholesky factorisation of a symmetric positive
    definite block A = L Lᵀ, with L lower-triangular (n×n). Now we want to form
    the Cholesky of the augmented block matrix:

        [ A   B ]      size (n+k) × (n+k)
        [ Bᵀ  C ]

    where B is (n×k) and C is (k×k). This corresponds to adding k new variables/
    observations in a Gaussian process setting.

    Parameters
    ----------
    cache : CholCache
        Holds the existing Cholesky factor L of the top-left block A.
    B : Array (n, k)
        Cross-covariance block between old and new variables.
    C : Array (k, k)
        Covariance block for the new variables.
    jitter : float, default=0.0
        Optional diagonal regularisation added to the Schur complement for
        numerical stability.

    Returns
    -------
    CholCache
        Updated cache containing the new (n+k) × (n+k) lower-triangular factor Lnew,
        such that

            [ A   B ] = Lnew Lnewᵀ
            [ Bᵀ  C ]

    Notes
    -----
    - The update proceeds by block Cholesky:
        1. Solve for W = L⁻¹ B.            (triangular solve)
        2. Compute the Schur complement S = C - Wᵀ W.
        3. Factorise S = L22 L22ᵀ.
        4. Assemble the extended factor:

               [ L    0 ]
        Lnew = [ Wᵀ  L22 ]

      This is cheaper (O(n k² + k³)) than recomputing the full Cholesky from scratch (O((n+k)³)).

    - The symmetrisation 0.5 * (S + S.T) ensures numerical symmetry before
      factorisation.
    """
    L = cache.L
    # Step 1: Solve for W = L⁻¹ B (triangular solve, O(n² k))
    W = jax.scipy.linalg.solve_triangular(L, B, lower=True, trans='N')

    # Step 2: Schur complement of C
    S = C - W.T @ W
    if jitter != 0.0:
        S = S + jitter * jnp.eye(S.shape[0], dtype=S.dtype)

    # Step 3: Factorise the Schur complement
    L22 = jnp.linalg.cholesky(0.5 * (S + S.T))

    # Step 4: Assemble the extended lower-triangular matrix
    n, k = L.shape[0], L22.shape[0]
    top = jnp.concatenate([L, jnp.zeros((n, k), dtype=L.dtype)], axis=1)
    bot = jnp.concatenate([W.T, L22], axis=1)
    Lnew = jnp.concatenate([top, bot], axis=0)

    return CholCache(Lnew)
