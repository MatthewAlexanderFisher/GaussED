from __future__ import annotations
from dataclasses import dataclass
from typing import Union    
from jax import Array
import jax
import jax.numpy as jnp

from gaussed.gp.gp_ops.base import KernelSpec, FunSpec, OpContext, Probe
from gaussed.linops import LinearOp
from gaussed.types import LinearLike

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class BasisRep:
    # we won’t use kernel_spec here; we use the basis explicitly
    kernel_spec: KernelSpec     # kept for interface completeness; unused
    mean_spec: FunSpec
    phi_spec: FunSpec           # φ: X -> (n, m)
    Lambda: Union[Array, float] # (m,m) or scalar

    def _design(self, F: Probe, ctx: OpContext) -> Array:
        # Materialise D for now; you can add a streaming version later
        return F.apply(self.phi_spec, ctx)   # (n, m)

    def gram(self, F: Probe, G: Probe, ctx: OpContext) -> LinearLike:
        # Only F==G used for training systems; for general cross you can materialise both
        if F is G:
            D = self._design(F, ctx)                             # (n, m)
            n, m = D.shape
            Lam = self.Lambda
            def mv(v: Array) -> Array:
                z = D.T @ v
                z = Lam * z if jnp.ndim(Lam)==0 else Lam @ z
                return D @ z
            return LinearOp((n, n), mv)
        # cross: dense block
        DF = self._design(F, ctx); DG = self._design(G, ctx)
        Lam = self.Lambda
        K = DF @ (Lam * DG.T if jnp.ndim(Lam)==0 else (Lam @ DG.T))
        return K

    def cross(self, F: Probe, G: Probe, ctx: OpContext) -> Array:
        DF = self._design(F, ctx); DG = self._design(G, ctx)
        Lam = self.Lambda
        return DF @ (Lam * DG.T if jnp.ndim(Lam)==0 else (Lam @ DG.T))

    def mean(self, F: Probe, ctx: OpContext) -> Array:
        return F.apply(self.mean_spec, ctx)[:, 0]
