from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
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
