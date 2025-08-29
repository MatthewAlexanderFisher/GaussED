from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Optional, Tuple, Callable, cast
from jax import Array
import jax.numpy as jnp
import jax

from gaussed.gp.gp_ops.base import Functional, FunSpec, KernelSpec, OpContext
from gaussed.domains.base import Domain

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Eval:
    X: Array  # (n_x, d)

    # Unary: realise g at X
    def __call__(self, g: FunSpec, ctx: OpContext) -> Array:
        G = g.eval(self.X)                 # (n_L, n_x) or (n_x,)
        return G if G.ndim == 2 else G[:, None]

    # Left kernel map: FunSpec of Y with maybe-analytic ∫_Y
    def left_kernel_map(self, ks: KernelSpec, ctx: OpContext) -> FunSpec:
        X = self.X

        def f_eval(Y: Array) -> Array:     # (q_y,d) -> (n_x, q_y)
            return ks.__call__(X, Y)

        # Maybe-analytic integral over Y: integrate_y_of(domY) -> Optional[EvalFn]
        f_integrate: Optional[Callable[[Domain], Optional[Array]]] = None
        if ks.integrate_y_of is not None:
            iy_of = ks.integrate_y_of      # Callable[[Domain], Optional[Callable[[Array], Array]]]

            def _integ(domY: Domain, iy_of=iy_of, X=X) -> Optional[Array]:
                Fy_opt = iy_of(domY)       # Optional[Callable[[Array], Array]]
                if Fy_opt is None:
                    return None
                return Fy_opt(X)           # (n_x, n_R)

            f_integrate = _integ

        # Derivatives wrt Y if available (signatures single-arg for eval + extra indices)
        if ks.d_dy is not None:
            h = cast(Callable[[Array, Array, int], Array], ks.d_dy)
            f_partial = lambda Y, j, h=h, X=X: h(X, Y, j)
        else:
            f_partial = None

        if ks.d2_yy is not None:
            h2 = cast(Callable[[Array, Array, int, int], Array], ks.d2_yy)
            f_partial2 = lambda Y, i, j, h2=h2, X=X: h2(X, Y, i, j)
        else:
            f_partial2 = None

        # FunSpec.integrate has type: Optional[Callable[[Domain], Optional[Array]]]
        return FunSpec(eval=f_eval, integrate=f_integrate, partial=f_partial, partial2=f_partial2)

    # Right reduction: apply Eval to a FunSpec of Y (evaluate at Y = X)
    def right_reduce(self, F: FunSpec, ctx: OpContext) -> Array:
        return F.eval(self.X)  # (n_L, n_R = n_x)

    # Pair: canonical composition
    def pair(self, other: Functional, ks: KernelSpec, ctx: OpContext) -> Array:
        F = self.left_kernel_map(ks, ctx)  # FunSpec(Y)
        return other.right_reduce(F, ctx)


    def tree_flatten(self): return ((self.X,), ())
    @classmethod
    def tree_unflatten(cls, aux, ch): (X,) = ch; return cls(X)




