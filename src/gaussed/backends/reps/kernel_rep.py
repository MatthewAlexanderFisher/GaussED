from __future__ import annotations
from dataclasses import dataclass, field
from jax import Array
import jax

from gaussed.gp.gp_ops.base import KernelSpec, FunSpec, OpContext
from gaussed.gp.gp_ops.probe import Probe, ProbeStack
from gaussed.linops.constructors import LinOpConstructor, DenseGramConstructor
from gaussed.types import LinearLike

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class KernelRep:
    _kernel_spec: KernelSpec
    _mean_spec: FunSpec
    ctx: OpContext
    linop_constructor: LinOpConstructor = field(default_factory=DenseGramConstructor)  

    @property
    def kernel_spec(self) -> KernelSpec:
        return self._kernel_spec

    @property
    def mean_spec(self) -> FunSpec:
        return self._mean_spec

    def gram(self, F: ProbeStack, G: ProbeStack, ctx: OpContext) -> LinearLike:
        linop = self.linop_constructor(self._kernel_spec, F, G, ctx)
        return linop

    def mean(self, F: ProbeStack, ctx: OpContext) -> LinearLike:
        return F.apply(self._mean_spec, ctx)

    # pytree (treat callables as aux if needed)
    def tree_flatten(self):
        children = (self._kernel_spec, self._mean_spec)
        aux = (self.ctx,)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        ks, ms = children
        return cls(ks, ms, aux)
