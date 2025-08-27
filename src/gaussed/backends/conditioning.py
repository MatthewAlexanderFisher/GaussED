from jax import Array
import jax.numpy as jnp

from gaussed.linops import LinearOp
from gaussed.types import LinearLike
from gaussed.gp.gp_ops.base import Probe
from gaussed.backends.base import Backend
from gaussed.model import GPModel


def condition(backend: Backend, model: "GPModel", F: Probe, y: Array, lik: "Likelihood"):
    # 1) choose representation
    rep = backend.make_rep(model)

    # 2) build training system A = K_FF + S (S = likelihood op)
    KFF = rep.gram(F, F, model.gp.backend.make_op())   # LinearLike (Array or LinearOp)
    S   = lik.op_for(F, dtype=y.dtype)   # must return LinearOp (or Array -> wrap)
    A   = add_linearlikes(KFF, S)        # small helper to sum Array/Op uniformly

    # 3) factor + solve
    fac = backend.factor(A)
    resid = y - rep.mean(F)              # (n,)
    alpha = fac.solve(resid)             # (n,)

    # 4) posterior handle
    return PosteriorGP(rep, F, fac, alpha, mean_fn=model.gp.mean, lik=lik)

# helper: sum LinearLike
def add_linearlikes(A: LinearLike, B: LinearLike) -> LinearLike:
    from gaussed.linops import LinearOp
    # both dense -> dense
    import jax.numpy as jnp
    if not isinstance(A, LinearOp) and not isinstance(B, LinearOp):
        return A + B
    # wrap dense as LinearOp if needed
    def to_op(X):
        if isinstance(X, LinearOp): return X
        n = X.shape[0]
        return LinearOp((n, n), mv=lambda v: X @ v)
    Aop, Bop = to_op(A), to_op(B)
    n = Aop.shape[0]
    return LinearOp((n, n), mv=lambda v: Aop.mv(v) + Bop.mv(v))
