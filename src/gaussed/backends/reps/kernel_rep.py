from __future__ import annotations
from dataclasses import dataclass
from jax import Array
import jax

from gaussed.gp.gp_ops.base import KernelSpec, FunSpec, OpContext
from gaussed.gp.gp_ops.probe import Probe, ProbeStack

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class KernelRep:
    _kernel_spec: KernelSpec
    _mean_spec: FunSpec
    ctx: OpContext

    # read-only properties to satisfy CovRep
    @property
    def kernel_spec(self) -> KernelSpec:
        return self._kernel_spec

    @property
    def mean_spec(self) -> FunSpec:
        return self._mean_spec

    def gram(self, F: ProbeStack, G: ProbeStack, ctx: OpContext) -> Array:
        return F.kernel(G, self._kernel_spec, ctx)

    def cross(self, F: ProbeStack, G: ProbeStack, ctx: OpContext) -> Array:
        return F.kernel(G, self._kernel_spec, ctx)

    def mean(self, F: ProbeStack, ctx: OpContext) -> Array:
        return F.apply(self._mean_spec, ctx)[:, 0]

    # pytree (treat callables as aux if needed)
    def tree_flatten(self):
        children = (self._kernel_spec, self._mean_spec)
        aux = (self.ctx,)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        ks, ms = children
        return cls(ks, ms, aux)
