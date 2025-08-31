from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Any, Tuple, cast, Protocol
import jax
import jax.numpy as jnp
from jax import Array

from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from gaussed.backends.base import Backend


from gaussed.gp.base import GP, PosteriorGP
from gaussed.likelihoods.base import Likelihood
from gaussed.likelihoods.gaussian import GaussianLikelihood
from gaussed.types import ProbeLike
from gaussed.gp.gp_ops.probe import Probe, ProbeStack, as_stack
from gaussed.backends.solvers.linear_solver import LinearSolver
from gaussed.linops.linop import LinearOp, AsLinearOp, materialise_dense
from gaussed.utils.shape_helpers import pack_vec


@jax.tree_util.register_pytree_node_class
@dataclass
class GPSample:
    """
    A JAX-traceable representation of a single GP sample path.

    The 'rep' encodes the sample in any backend-specific latent basis
    (e.g., weights for features, or a draw of the posterior coefficients).
    The 'eval_fn' is a *static* function of signature:
        eval_fn(rep: Array, X: Array) -> Array
    returning f_sample(X) with shape (n, output_dim) or (n,).
    """
    rep: Array
    eval_fn: Callable[[Array, Array], Array] = field(repr=False, default=lambda r, x: x)

    def __call__(self, X: Array) -> Array:
        return self.eval_fn(self.rep, X)

    # ---- pytree plumbing ----
    def tree_flatten(self):
        children = (self.rep,)
        aux = (self.eval_fn,)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (eval_fn,) = aux
        (rep,) = children
        return cls(rep=rep, eval_fn=eval_fn)

    @classmethod
    def axes(cls, rep_axis: Array):
        obj = object.__new__(cls)
        obj.rep = rep_axis
        obj.eval_fn = lambda r, x: x  # static placeholder; unused in axis objects
        return obj


@jax.tree_util.register_pytree_node_class
@dataclass
class PosteriorState:
    """
    Thin, JIT-safe wrapper around your GP posterior.

    You provide three *static* backend hooks:
        predict_fn(params, X) -> (mean, var)
        sample_fn(params, key) -> GPSample
        # optional, only needed if you want an in-JAX update step:
        update_fn(params, x_new, y_new) -> params_new

    'params' is the only dynamic child (PyTree). All callables are static.
    """
    params: Any
    predict_fn: Callable[[Any, Array], Tuple[Array, Array]] = field(repr=False, default=lambda p, x: (x, x))
    sample_fn: Callable[[Any, Array], GPSample] = field(repr=False, default=lambda p, k: GPSample(jnp.zeros(())))
    update_fn: Optional[Callable[[Any, Array, Array], Any]] = field(repr=False, default=None)

    def predict(self, X: Array) -> Tuple[Array, Array]:
        return self.predict_fn(self.params, X)

    def sample(self, key: Array) -> GPSample:
        return self.sample_fn(self.params, key)

    def update(self, x_new: Array, y_new: Array) -> "PosteriorState":
        if self.update_fn is None:
            # Return self unchanged if no updater provided (by design, avoid Python exceptions in jit)
            return self
        new_params = self.update_fn(self.params, x_new, y_new)
        return PosteriorState(new_params, self.predict_fn, self.sample_fn, self.update_fn)

    # ---- pytree plumbing ----
    def tree_flatten(self):
        children = (self.params,)
        aux = (self.predict_fn, self.sample_fn, self.update_fn)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        predict_fn, sample_fn, update_fn = aux
        (params,) = children
        return cls(params, predict_fn, sample_fn, update_fn)

    @classmethod
    def axes(cls, params_axis: Any):
        obj = object.__new__(cls)
        obj.params = params_axis
        obj.predict_fn = lambda p, x: (x, x)
        obj.sample_fn = lambda p, k: GPSample(jnp.zeros(()))
        obj.update_fn = None
        return obj

# ===============================
#       Acquisition context
# ===============================

@dataclass
class AcqContext:
    """
    Lightweight runtime data for acquisitions.
    All fields are *arguments* (not part of pytrees), so they can be traced.

    y_best: scalar (maximisation) or per-output best; shape () or (out_dim,)
    key: PRNG key for stochastic acquisitions (e.g., Thompson)
    """
    y_best: Optional[Array] = None
    key: Optional[Array] = None
    step: Optional[Array] = None  # jnp.int32 scalar if you want time-varying schedules


# ===============================
#       Acquisition base + impls
# ===============================

class Acquisition(Protocol):
    """
    Base callable: __call__(X, posterior, ctx) -> utility per-candidate.
    Subclasses only carry (PyTree) params as children; behaviour is static.
    """
    def __call__(self, X: Array, posterior: PosteriorState, ctx: AcqContext) -> Array:
        raise NotImplementedError("Implement in subclasses.")


# -------- UCB --------
@jax.tree_util.register_pytree_node_class
@dataclass
class UCB(Acquisition):
    beta: Array  # can be scalar or schedule value passed/frozen into the object

    def __call__(self, X: Array, posterior: PosteriorState, ctx: AcqContext) -> Array:
        m, v = posterior.predict(X)
        s = jnp.sqrt(jnp.maximum(v, 0.0))
        return m + jnp.sqrt(self.beta) * s

    def tree_flatten(self):
        return (self.beta,), ()

    @classmethod
    def tree_unflatten(cls, aux, children):
        (beta,) = children
        return cls(beta)

    @classmethod
    def axes(cls, beta_axis: Array):
        obj = object.__new__(cls)
        obj.beta = beta_axis
        return obj


# ---- helpers for EI ----
def _std_normal_pdf(z: Array) -> Array:
    return jnp.exp(-0.5 * z * z) / jnp.sqrt(2.0 * jnp.pi)

def _std_normal_cdf(z: Array) -> Array:
    # 0.5 * [1 + erf(z / sqrt(2))]
    return 0.5 * (1.0 + jax.lax.erf(z / jnp.sqrt(2.0)))


# -------- EI --------
@jax.tree_util.register_pytree_node_class
@dataclass
class ExpectedImprovement(Acquisition):
    xi: Array = field(default_factory=lambda: jnp.array(0.0))

    def __call__(self, X: Array, posterior: PosteriorState, ctx: AcqContext) -> Array:
        m, v = posterior.predict(X)
        s = jnp.sqrt(jnp.maximum(v, 0.0))
        # Robust z: if s==0, define EI=0
        yb = jnp.asarray(ctx.y_best) if ctx.y_best is not None else jnp.max(m)
        z = jnp.where(s > 0, (m - yb - self.xi) / s, jnp.zeros_like(m))
        ei = (m - yb - self.xi) * _std_normal_cdf(z) + s * _std_normal_pdf(z)
        return jnp.where(s > 0, ei, jnp.zeros_like(ei))

    def tree_flatten(self):
        return (self.xi,), ()

    @classmethod
    def tree_unflatten(cls, aux, children):
        (xi,) = children
        return cls(xi)

    @classmethod
    def axes(cls, xi_axis: Array):
        obj = object.__new__(cls)
        obj.xi = xi_axis
        return obj


# -------- Thompson Sampling (utility = sampled f) --------
@jax.tree_util.register_pytree_node_class
@dataclass
class Thompson(Acquisition):
    """
    Thompson acquisition: draw a GP sample and return f_sample(X).
    The draw is done inside 'posterior.sample(ctx.key)'.
    """
    jitter: Array = field(default_factory=lambda: jnp.array(0.0))

    def __call__(self, X: Array, posterior: PosteriorState, ctx: AcqContext) -> Array:
        key = ctx.key if ctx.key is not None else jax.random.PRNGKey(0)
        f = posterior.sample(key)
        vals = f(X)
        if self.jitter is not None:
            vals = vals + self.jitter * jax.random.normal(key, shape=vals.shape)
        return vals

    def tree_flatten(self):
        return (self.jitter,), ()

    @classmethod
    def tree_unflatten(cls, aux, children):
        (jitter,) = children
        return cls(jitter)

    @classmethod
    def axes(cls, jitter_axis: Array):
        obj = object.__new__(cls)
        obj.jitter = jitter_axis
        return obj


# ===============================
#       Candidate proposers
# ===============================

@jax.tree_util.register_pytree_node_class
@dataclass
class FixedSetProposer:
    """Always proposes a fixed candidate set Xcand of shape (n, d)."""
    Xcand: Array

    def __call__(self, key: Array) -> Array:
        return self.Xcand

    def tree_flatten(self):
        return (self.Xcand,), ()

    @classmethod
    def tree_unflatten(cls, aux, children):
        (Xcand,) = children
        return cls(Xcand)

    @classmethod
    def axes(cls, Xcand_axis: Array):
        obj = object.__new__(cls)
        obj.Xcand = Xcand_axis
        return obj


@jax.tree_util.register_pytree_node_class
@dataclass
class RandomBoxProposer:
    """
    Uniformly samples 'n_candidates' points in a box [lo, hi].
    lo, hi: shape (d,)
    """
    lo: Array
    hi: Array
    n_candidates: int

    def __call__(self, key: Array) -> Array:
        u = jax.random.uniform(key, shape=(self.n_candidates, self.lo.shape[0]))
        return self.lo + (self.hi - self.lo) * u

    def tree_flatten(self):
        return (self.lo, self.hi, jnp.array(self.n_candidates)), ()

    @classmethod
    def tree_unflatten(cls, aux, children):
        lo, hi, n = children
        return cls(lo, hi, int(n.item()))

    @classmethod
    def axes(cls, lo_axis: Array, hi_axis: Array, n_axis: Array):
        obj = object.__new__(cls)
        obj.lo = lo_axis
        obj.hi = hi_axis
        obj.n_candidates = int(n_axis.item())
        return obj


# ===============================
#          SED policies
# ===============================

@jax.tree_util.register_pytree_node_class
@dataclass
class GreedyArgmaxPolicy:
    """
    1) Build candidate set via proposer(key)
    2) Score via acquisition(Xcand, posterior, ctx)
    3) Return argmax and extras.

    For batch via Thompson: sample once and take top-k of the same sample.
    """
    acq: Acquisition
    proposer: Any  # Callable[[key], Xcand]; kept static in aux if needed

    def propose_one(self, posterior: PosteriorState, key: Array, ctx: AcqContext
                    ) -> Tuple[Array, dict]:
        k1, k2 = jax.random.split(key)
        Xcand = self.proposer(k1)
        util = self.acq(Xcand, posterior, AcqContext(y_best=ctx.y_best, key=k2, step=ctx.step))
        idx = jnp.argmax(util.reshape(-1))
        x_next = Xcand[idx]
        info = {"Xcand": Xcand, "util": util, "idx": idx}
        return x_next, info

    def propose_topk_thompson(self, posterior: PosteriorState, key: Array, k: int,
                              Xcand: Optional[Array] = None) -> Tuple[Array, dict]:
        """
        Single Thompson draw → select top-k.
        If Xcand is None, it is sampled from proposer(key).
        """
        k1, k2 = jax.random.split(key)
        if Xcand is None:
            Xcand = self.proposer(k1)

        # one sample path; reuse across all candidates
        f = posterior.sample(k2)
        vals = f(cast(Array, Xcand))  # shape (n,)
        # top-k indices (stable): argsort descending then take first k
        order = jnp.argsort(-vals.reshape(-1))
        idxs = order[:k]
        Xk = cast(Array, Xcand)[idxs]
        info = {"Xcand": Xcand, "sample_vals": vals, "idxs": idxs}
        return Xk, info

    # ---- pytree plumbing ----
    def tree_flatten(self):
        # acq is child (dynamic); proposer is static (aux) since it can contain callables
        children = (self.acq,)
        aux = (self.proposer,)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (proposer,) = aux
        (acq,) = children
        return cls(acq, proposer)

    @classmethod
    def axes(cls, acq_axis: Acquisition):
        obj = object.__new__(cls)
        obj.acq = acq_axis
        obj.proposer = lambda k: jnp.zeros((1, 1))  # unused placeholder
        return obj


# ===============================
#       One-step helpers (JIT)
# ===============================

def propose_next(
    policy: GreedyArgmaxPolicy,
    posterior: PosteriorState,
    key: Array,
    y_best: Optional[Array] = None,
    step: Optional[int] = None,
) -> Tuple[Array, dict]:
    """
    Propose a single next design point (greedy argmax).
    JIT-safe, no growing arrays.
    """
    ctx = AcqContext(y_best=y_best, key=None, step=(None if step is None else jnp.array(step, jnp.int32)))
    return policy.propose_one(posterior, key, ctx)


def propose_batch_thompson(
    policy: GreedyArgmaxPolicy,
    posterior: PosteriorState,
    key: Array,
    batch_size: int,
    Xcand: Optional[Array] = None,
) -> Tuple[Array, dict]:
    """
    Batch selection via a *single* Thompson draw → top-k.
    Avoids fantasy updates; fully JIT/table-stakes.
    """
    return policy.propose_topk_thompson(posterior, key, batch_size, Xcand)
