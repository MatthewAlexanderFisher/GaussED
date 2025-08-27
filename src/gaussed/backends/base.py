from __future__ import annotations
from typing import Protocol, Tuple
from dataclasses import dataclass
from jax import Array
import jax


from gaussed.types import LinearLike
from gaussed.gp.gp_ops.base import KernelSpec, FunSpec, OpContext
from gaussed.model import GPModel
from gaussed.backends.reps.base import CovRep
from gaussed.backends.solvers.base import Factor


# Base Backend
class Backend(Protocol):
    name: str

    def make_specs(self, model: "GPModel") -> Tuple[KernelSpec, FunSpec, OpContext]:
        ...

    def make_rep(self, model: "GPModel") -> CovRep:
        ...

    def factor(self, A: LinearLike) -> Factor:
        ...

@dataclass
class ExactBackend(Backend):
    name: str = "exact"

    def make_specs(self, model: "GPModel"):
        mean_spec: FunSpec   = model.gp.mean.to_spec()
        kernel_spec: KernelSpec = model.gp.kernel.to_spec(model.domain)
        ctx = OpContext(quad_rule=model.quad_rule)  # optional
        return kernel_spec, mean_spec, ctx

    def make_rep(self, model: "GPModel"):
        ks, ms, ctx = self.make_specs(model)
        # stash ctx on model (optional convenience)
        model.op_ctx = ctx
        return KernelRep(kernel_spec=ks, mean_spec=ms)
