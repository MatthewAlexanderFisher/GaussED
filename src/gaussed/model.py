from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Callable, Any, Union
import jax
import jax.numpy as jnp
from jax import Array

from gaussed.gp.base import GP, PosteriorGP
from gaussed.backends import Backend
from gaussed.engines.likelihoods.base import Likelihood
from gaussed.engines.likelihoods.gaussian import GaussianLikelihood
from gaussed.types import ProbeLike

@dataclass
class GPModel:
    gp: GP
    likelihood: Likelihood = GaussianLikelihood()   # Gaussian, Bernoulli, Poisson, Student-t, ...

    def condition(self, F: ProbeLike, y: Array, *, backend: "Backend") -> PosteriorGP:
        F = self.gp._as_probe(F)
        K_FF = self.gp.K(F, F)
        mu_F = F.mean(self.gp.mean)
        if isinstance(self.likelihood, GaussianLikelihood):
            Sigma = jnp.asarray(self.likelihood.Sigma_for(F, dtype=K_FF.dtype))
            cond = backend.condition(K_FF, Sigma)
            alpha = cond.solve(y - mu_F)
            return PosteriorGP(self.gp, F, cond, alpha)
        else:
            # defer to non-Gaussian engine (Laplace/EP/VI)
            return self.likelihood.laplace_or_ep_condition(self.gp, F, y, backend)
