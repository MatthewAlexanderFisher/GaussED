from __future__ import annotations
from dataclasses import dataclass, field
from typing import Union, Optional
from jax import Array
import jax
import jax.numpy as jnp

from gaussed.gp.gp_ops.base import KernelSpec, BasisSpec, FunSpec, OpContext
from gaussed.gp.gp_ops.probe import ProbeStack
from gaussed.linops.constructors import LinOpConstructor, ColaConstructor, DenseGramConstructor
from gaussed.types import LinearLike



# ---- BasisRep ----
@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class BasisRep:
    kernel_spec: KernelSpec                 # kept for API parity (unused)
    mean_spec: FunSpec
    phi_spec: Union[BasisSpec, FunSpec]     # if FunSpec, we’ll wrap it
    Lambda: Array                           # scalar, (m,), or (m,m)
    linop_constructor: LinOpConstructor = field(default_factory=DenseGramConstructor)

    def _as_basis_spec(self) -> BasisSpec:
        phi = self.phi_spec if isinstance(self.phi_spec, FunSpec) else self.phi_spec.phi_spec
        return BasisSpec(phi_spec=phi, Lambda=self.Lambda)

    def gram(self, F: ProbeStack, G: ProbeStack, ctx: OpContext) -> LinearLike:
        bs = BasisSpec(phi_spec=self.as_phi_spec(self.phi_spec), Lambda=self.Lambda)
        return self.linop_constructor(bs, F, G, ctx)


    def mean(self, F: ProbeStack, ctx: OpContext) -> Array:
        mu = F.apply(self.mean_spec, ctx)
        return mu[:, 0] if mu.ndim == 2 and mu.shape[1] == 1 else mu

    def tree_flatten(self):
        children = (self.Lambda,)
        aux = (self.kernel_spec, self.mean_spec, self.phi_spec, self.linop_constructor)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (Lam,) = children
        kernel_spec, mean_spec, phi_spec, linop_constructor = aux
        return cls(kernel_spec, mean_spec, phi_spec, Lam, linop_constructor)

    def as_phi_spec(self, obj: Union[FunSpec, BasisSpec]) -> FunSpec:
        return obj if isinstance(obj, FunSpec) else obj.phi_spec

