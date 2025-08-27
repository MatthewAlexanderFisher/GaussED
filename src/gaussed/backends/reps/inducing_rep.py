from __future__ import annotations
from dataclasses import dataclass
from jax import Array
import jax.numpy as jnp
import jax

from gaussed.gp.gp_ops.base import KernelSpec, FunSpec, OpContext, Probe
from gaussed.types import LinearLike
from gaussed.gp.gp_ops.point_eval import Eval
from gaussed.linops import LinearOp

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class InducingRep:
    kernel_spec: KernelSpec
    mean_spec: FunSpec
    Z: Array                  # (m, d)
    jitter: float = 1e-6

    # cached factors could be added; for clarity we factor on demand here
    def _Kuu(self) -> Array:
        Z = self.Z
        return self.kernel_spec.k0(Z, Z) + self.jitter * jnp.eye(Z.shape[0], dtype=Z.dtype)

    def _Kfu(self, F: Probe, ctx: OpContext) -> Array:
        # left: F probe; right: Eval at inducing Z
        U = Probe(ops=(), fnl=Eval(self.Z))
        return F.kernel(U, self.kernel_spec, ctx)          # (n, m)

    def gram(self, F: Probe, G: Probe, ctx: OpContext) -> LinearLike:
        # training: usually F is G, so we deliver a LinearOp y = Q_FF v
        if F is G:
            Kfu = self._Kfu(F, ctx)                        # (n, m)
            Kuu = self._Kuu()                              # (m, m)
            # precompute solve operator in mv; for speed, factor once outside if you like
            Kuu_fac = jnp.linalg.cholesky(Kuu)
            n, m = Kfu.shape
            def mv(v: Array) -> Array:
                z = Kfu.T @ v                              # (m,)
                # solve Kuu w = z
                w = jax.scipy.linalg.cho_solve((Kuu_fac, True), z)
                return Kfu @ w                             # (n,)
            return LinearOp((Kfu.shape[0], Kfu.shape[0]), mv)

        # general cross Q_FG = K_FU K_UU^{-1} K_UG  (dense)
        Kfu = self._Kfu(F, ctx)                            # (nF, m)
        Kug = self._Kfu(G, ctx).T                          # (m, nG) since K_GU = K_UG^T
        Kuu = self._Kuu()                                  # (m, m)
        Kuu_fac = jnp.linalg.cholesky(Kuu)
        Kuu_inv_Kug = jax.scipy.linalg.cho_solve((Kuu_fac, True), Kug)  # (m, nG)
        return Kfu @ Kuu_inv_Kug                           # (nF, nG)

    def cross(self, F: Probe, G: Probe, ctx: OpContext) -> LinearLike:
        return self.gram(F, G, ctx)  # returns dense in cross case

    def mean(self, F: Probe, ctx: OpContext) -> Array:
        return F.apply(self.mean_spec, ctx)[:, 0]
