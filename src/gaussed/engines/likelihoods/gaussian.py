from __future__ import annotations
from typing import Protocol, Optional, Callable
from dataclasses import dataclass, field
from jax import Array
import jax
import jax.numpy as jnp 

from gaussed.gp.gp_ops.base import Probe
from gaussed.engines.likelihood.gaussian_noise import NoiseSpec, Noiseless
from gaussed.gp.base import GP, PosteriorGP
from gaussed.engines.backends.base import Backend
from gaussed.engines.linops import LinearOp, SumOp
from gaussed.types import ProbeLike

@jax.tree_util.register_pytree_node_class
@dataclass
class GaussianLikelihood:
    """Gaussian observation model, parameterised by a NoiseSpec (operator-first)."""
    noise: NoiseSpec = field(default_factory=Noiseless)

    # ---------- Operator-first API ----------
    def op_for(self, F: Probe, dtype) -> LinearOp:
        """Return Σ(F) as a LinearOp. Uses sizes from F if the noise needs them (e.g. BlockNoise)."""
        # Prefer a probe-aware constructor if available (e.g. BlockNoise.as_op_for_probe)
        if hasattr(self.noise, "as_op_for_probe"):
            return self.noise.as_op_for_probe(F, dtype)
        
        return self.noise.as_op(F.n(), dtype)

    def gram_op(self, K_op: LinearOp, F: Probe, dtype) -> LinearOp:
        """Return (K_FF + Σ) as a LinearOp (for CG/matrix-free backends)."""
        return SumOp(K_op, self.op_for(F, dtype))

    # ---------- Dense convenience (for exact Cholesky/variational) ----------
    def Sigma_for(self, F: Probe, dtype) -> Array:
        """Materialise Σ(F) as a dense matrix."""
        return self.op_for(F, dtype).to_dense()

    def add_to_gram(self, K_FF: Array, F: Probe) -> Array:
        """Return K_FF + Σ(F) as a dense matrix."""
        return K_FF + self.Sigma_for(F, dtype=K_FF.dtype)

    def laplace_or_ep_condition(self, gp: GP, F: ProbeLike, y: Array, backend: Backend) -> PosteriorGP:
        raise NotImplementedError("Non-Gaussian likelihoods must implement laplace_or_ep_condition.")

    # ---------- Optional: direct mv helper (rarely needed) ----------
    def mv(self, v: Array, F: Probe) -> Array:
        """Apply Σ(F) to a vector v (handy in bespoke CG code)."""
        return self.op_for(F, v.dtype).mv(v)

    # ---------- PyTree plumbing ----------
    def tree_flatten(self):
        return (self.noise,), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        (noise,) = children
        return cls(noise)

    @classmethod
    def axes(cls, noise_axis):
        obj = object.__new__(cls)
        obj.noise = noise_axis
        return obj
