from __future__ import annotations
from typing import Protocol
import jax, jax.numpy as jnp
from dataclasses import dataclass
from jax import Array

class Transform(Protocol):
    def forward(self, x: Array) -> Array: ...
    def inverse(self, y: Array) -> Array: ...
    def log_abs_det_jacobian(self, x: Array) -> Array: ...

@dataclass
class Positive(Transform):
    eps: float = 1e-12
    def forward(self, x):  return jax.nn.softplus(x) + self.eps
    def inverse(self, y):  return jnp.log(jnp.expm1(jnp.maximum(y - self.eps, self.eps)))
    def log_abs_det_jacobian(self, x):  # log d/ dx softplus(x)
        return -jnp.log1p(jnp.exp(-x))  # = log(sigmoid(x))

@dataclass
class Bounded(Transform):
    low: float
    high: float
    eps: float = 1e-12

    def forward(self, x):
        s = jax.nn.sigmoid(x)
        return self.low + (self.high - self.low) * (s * (1 - 2*self.eps) + self.eps)
    
    def inverse(self, y):
        a = (y - self.low) / (self.high - self.low)
        a = jnp.clip(a, self.eps, 1 - self.eps)
        return jnp.log(a) - jnp.log1p(-a)
    
    def log_abs_det_jacobian(self, x):
        # log((high-low) * sigmoid(x)*(1-sigmoid(x)))
        s = jax.nn.sigmoid(x)
        return jnp.log(self.high - self.low) + jnp.log(s) + jnp.log1p(-s)

@dataclass
class Simplex(Transform):
    axis: int = -1
    def forward(self, x): return jax.nn.softmax(x, axis=self.axis)
    def inverse(self, y):
        # softmax inverse up to additive constant – fix last coord to 0
        z = jnp.log(y) - jnp.expand_dims(jnp.log(y[..., -1]), axis=self.axis)
        return z[..., :-1]
    def log_abs_det_jacobian(self, x):
        # For MCMC in unconstrained space, often not needed explicitly; omit or implement if required.
        raise NotImplementedError

@dataclass
class TriLPositive(Transform):
    """Unconstrained vector u -> lower-triangular L with positive diagonal."""
    dim: int
    eps: float = 1e-12

    def forward(self, x: Array) -> Array:
        """u[..., n] -> L[..., d, d], n=d(d+1)/2, diag softplus'ed."""
        d = self.dim
        n = d*(d+1)//2
        assert x.shape[-1] == n
        L = jnp.zeros((*x.shape[:-1], d, d), dtype=x.dtype)
        idx = 0
        for i in range(d):
            # fill row i, cols 0..i
            L = L.at[..., i, :i+1].set(x[..., idx:idx+i+1])
            idx += i + 1
        diag_raw = jnp.diagonal(L, axis1=-2, axis2=-1)
        diag_pos = jax.nn.softplus(diag_raw) + self.eps
        L = L - jnp.diagflat(diag_raw) + jnp.diagflat(diag_pos)
        return L

    def inverse(self, y: Array) -> Array:
        """Pack lower-tri L (with positive diag) back to unconstrained u."""
        d = self.dim
        diag = jnp.diagonal(y, axis1=-2, axis2=-1)
        diag_raw = jnp.log(jnp.expm1(jnp.maximum(diag - self.eps, self.eps)))
        L = y - jnp.diagflat(diag) + jnp.diagflat(diag_raw)
        parts = [L[..., i, :i+1] for i in range(d)]
        return jnp.concatenate(parts, axis=-1)

    def log_abs_det_jacobian(self, x: Array) -> Array:
        """log|∂Σ/∂u| where Σ = L(u) L(u)^T."""
        d = self.dim
        # grab diag_raw entries from u: positions idx+i in the packing loop
        diag_raw = []
        idx = 0
        for i in range(d):
            diag_raw.append(x[..., idx + i])
            idx += i + 1
        diag_raw = jnp.stack(diag_raw, axis=-1)
        diag_pos = jax.nn.softplus(diag_raw) + self.eps
        # L -> Σ Jacobian volume (standard result) + softplus diag contribution
        logJ_L_to_S = d * jnp.log(2.0) + jnp.sum(jnp.arange(d, 0, -1) * jnp.log(diag_pos), axis=-1)
        logJ_softplus = jnp.sum(-jnp.log1p(jnp.exp(-diag_raw)), axis=-1)  # sum log(sigmoid)
        return logJ_L_to_S + logJ_softplus
