from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Callable, Any, Union
import jax
import jax.numpy as jnp
from jax import Array

from gaussed.engines.backends.base import Backend
from gaussed.gp.kernels.base import Kernel
from gaussed.gp.means import MeanFunction
from gaussed.domains.base import Domain
from gaussed.codomains.base import Codomain
from gaussed.gp.gp_ops import Probe, Eval
from gaussed.engines.backends.base import Conditioner
from gaussed.types import ProbeLike

@jax.tree_util.register_pytree_node_class
@dataclass(init=False)
class GP:
    domain: Domain
    codomain: Codomain
    mean: MeanFunction
    kernel: Kernel

    def __init__(self, domain: Domain, codomain: Codomain, mean: MeanFunction, kernel: Kernel):
        self.domain = domain
        self.codomain = codomain
        self.mean = mean
        self.kernel = kernel

    # --- covariance blocks via probes ---
    def K(self, A: ProbeLike, B: ProbeLike) -> Array:
        A = self._as_probe(A)
        B = self._as_probe(B)
        return A.K_with(B, self.kernel, self.domain)

    # Conditioning is GP-API specific - Likelihoods (e.g. noise via Gaussian) are handled via the backend call,
    def condition(self, F: ProbeLike, y: Array, *, backend: "Backend", noise: Any) -> "PosteriorGP":
        F = self._as_probe(F)
        K_FF = self.K(F, F)              # (nF, nF)
        mu_F = F.mean(self.mean)         # (nF,)
        cond = backend.condition(K_FF, noise)   # backend interprets `noise` (e.g., NoiseSpec)
        alpha = cond.solve(y - mu_F)     # (nF,)
        return PosteriorGP(self, F, cond, alpha)


    def tree_flatten(self):
        # domain, codomain, kernel are leaves, no aux
        return (self.domain, self.codomain, self.mean, self.kernel), None
    
    @classmethod
    def tree_unflatten(cls, aux, children):
        dom, cod, mean, ker = children
        return cls(dom, cod, mean, ker)


    @classmethod
    def axes(cls, dom_axis, cod_axis, mean_axis, ker_axis):
        obj = object.__new__(cls)
        obj.domain = dom_axis
        obj.codomain = cod_axis
        obj.mean = mean_axis
        obj.kernel = ker_axis
        return obj

    @staticmethod
    def _as_probe(x_or_probe) -> Probe:
        return x_or_probe if hasattr(x_or_probe, "K_with") else Eval(jnp.asarray(x_or_probe))

@jax.tree_util.register_pytree_node_class
@dataclass
class PosteriorGP:
    prior: GP
    F: Probe
    cond: "Conditioner"
    alpha: Array  # (nF,)

    def mean(self, Q: ProbeLike) -> Array:
        Q = self.prior._as_probe(Q)
        K_QF = Q.K_with(self.F, self.prior.kernel, self.prior.domain)  # (nQ, nF)
        mu_Q = Q.mean(self.prior.mean)                                  # (nQ,)
        return mu_Q + K_QF @ self.alpha

    def cov(self, Q: ProbeLike, Qp: ProbeLike) -> Array:
        Q = self.prior._as_probe(Q)
        Qp = self.prior._as_probe(Qp)
        K_QF  = Q.K_with(self.F, self.prior.kernel, self.prior.domain)   # (nQ, nF)
        K_FQp = self.F.K_with(Qp, self.prior.kernel, self.prior.domain)  # (nF, nQp)
        V = self.cond.solve_blocks(K_FQp)                                 # (nF, nQp)
        K_QQp = Q.K_with(Qp, self.prior.kernel, self.prior.domain)        # (nQ, nQp)
        return K_QQp - K_QF @ V

    def sample(self, key, Q: Probe, n: int = 1) -> Array:
        # Exact posterior sampling with Cholesky backend
        # 1) draw eps ~ N(0, I), map via L_Q: L_Q L_Q^T = cov(Q,Q)
        # 2) add mean
        K = self.cov(Q, Q)                            # (nQ, nQ)
        L = jnp.linalg.cholesky(K + 1e-6*jnp.eye(K.shape[0], K.dtype))
        eps = jax.random.normal(key, (n, K.shape[0]))
        return self.mean(Q)[None, :] + eps @ L.T

    def variance(self, Q: ProbeLike) -> Array:
        Q = self.prior._as_probe(Q)
        K_QF = Q.K_with(self.F, self.prior.kernel, self.prior.domain)     # (nQ, nF)
        V = self.cond.solve_blocks(self.F.K_with(Q, self.prior.kernel, self.prior.domain))  # (nF, nQ)
        k_diag = jnp.diag(Q.K_with(Q, self.prior.kernel, self.prior.domain))
        return k_diag - jnp.sum(K_QF * V.T, axis=1)

    def tree_flatten(self):
        return (self.prior, self.F, self.cond, self.alpha), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        prior, F, cond, alpha = children
        return cls(prior, F, cond, alpha)

    # (Optional) sampling with matvec-only backend: use Lanczos / features instead
