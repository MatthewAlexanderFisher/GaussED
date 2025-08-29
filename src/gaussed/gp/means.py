from __future__ import annotations
from dataclasses import dataclass
import jax, jax.numpy as jnp
from jax import Array
from typing import Tuple, Callable, Protocol, Optional

from gaussed.domains.base import Domain 
from gaussed.codomains.base import Codomain

class MeanFun(Protocol):


    @property
    def domain(self) -> Domain: ...

    @property
    def codomain(self) -> Codomain: ...


    def __call__(self, X: Array, domain: Domain) -> Array: ...  # → (n,)

@dataclass(frozen=True)
class ZeroMeanFun:
    _domain: "Domain"    # static shape info
    _codomain: "Codomain"

    @property
    def domain(self) -> Domain: return self._domain

    @property
    def codomain(self) -> Codomain: return self._codomain


    def __call__(self, X: Array, domain: "Domain") -> Array:
        """
        Parameters
        ----------
        X : Array
            Shape (n, *domain.input_shape).
        domain : Domain
            Passed in by framework; can be checked against self.domain.

        Returns
        -------
        Array
            Shape (n, *codomain.output_shape).
        """
        X = jnp.asarray(X)
        # batch size = n = first dim
        n = X.shape[0]
        out_shape = self.codomain.output_shape
        return jnp.zeros((n, *out_shape), dtype=X.dtype)
