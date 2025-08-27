from __future__ import annotations
from dataclasses import dataclass
from jax import Array
import jax

from gaussed.gp.gp_ops.base import KernelSpec, FunSpec, OpContext, Probe
from gaussed.types import LinearLike

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class KernelRep:
    kernel_spec: KernelSpec
    mean_spec: FunSpec

    # exact Gram; return dense; wrap as LinearOp upstream if desired
    def gram(self, F: Probe, G: Probe, ctx: OpContext) -> LinearLike:
        K = F.kernel(G, self.kernel_spec, ctx)     # (n_F, n_G) array
        return K

    def cross(self, F: Probe, G: Probe, ctx: OpContext) -> Array:
        return F.kernel(G, self.kernel_spec, ctx)

    def mean(self, F: Probe, ctx: OpContext) -> Array:
        return F.apply(self.mean_spec, ctx)[:, 0]
