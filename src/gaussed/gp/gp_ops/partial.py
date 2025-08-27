from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Optional, Tuple, Callable, cast
from jax import Array
import jax.numpy as jnp
import jax

from gaussed.gp.gp_ops.base import Operator, Functional, OpContext, FunSpec, KernelSpec
from gaussed.utils.diff_helpers import _dir_tangent_like, _partial_rows, _partial_rows_x, _partial_rows_y, _mixed_xy

# === Partial operator =========================================================

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Partial:
    axis: int

    # Unary lift on functions: FunSpec -> FunSpec
    def __call__(self, g: FunSpec) -> FunSpec:
        ax = self.axis
        if g.partial is not None:
            p1 = cast(Callable[[Array, int], Array], g.partial)
            ev = lambda X, p1=p1: p1(X, ax)  # (n,q)
            p2 = g.partial2
            nxt = (lambda X, j, p2=cast(Callable[[Array, int, int], Array], p2): p2(X, ax, j)) if p2 is not None else None
            return FunSpec(eval=ev, partial=nxt, partial2=None)

        # AD fallback via directional jvp: rowwise ∂/∂x_ax
        ev = _partial_rows(g.eval, ax)  # (n,q)
        return FunSpec(eval=ev, partial=None, partial2=None)

    # Kernel lifts (left/right): prefer analytic hooks; else AD via jvp
    def lift_left(self, ks: KernelSpec, ctx: OpContext) -> KernelSpec:
        ax = self.axis
        if ks.d_dx is not None:
            d_dx_hook = cast(Callable[[Array, Array, int], Array], ks.d_dx)
            k0p = lambda X, Y, h=d_dx_hook: h(X, Y, ax)

            d2_xx_hook = ks.d2_xx
            d2_xy_hook = ks.d2_xy
            d_dx_next: Optional[Callable[[Array, Array, int], Array]] = (
                (lambda X, Y, i, h=cast(Callable[[Array, Array, int, int], Array], d2_xx_hook): h(X, Y, ax, i))
                if d2_xx_hook is not None else None
            )
            d_dy_next: Optional[Callable[[Array, Array, int], Array]] = (
                (lambda X, Y, j, h=cast(Callable[[Array, Array, int, int], Array], d2_xy_hook): h(X, Y, ax, j))
                if d2_xy_hook is not None else None
            )
            return KernelSpec(k0=k0p, d_dx=d_dx_next, d_dy=d_dy_next,
                              d2_xx=None, d2_yy=None, d2_xy=None,
                              integrate_x=ks.integrate_x, integrate_y=ks.integrate_y, integrate_xy=ks.integrate_xy)

        # AD fallback
        k0p = _partial_rows_x(ks.k0, ax)  # ∂/∂x_ax k
        # Offer mixed x–y derivatives so a future right-lift can reuse them cheaply
        def d_dy(X: Array, Y: Array, j: int) -> Array:
            _, dy = jax.jvp(lambda Y_: k0p(X, Y_), (Y,), (_dir_tangent_like(Y, j),))
            return dy  # (n_x, n_y)

        return KernelSpec(k0=k0p, d_dy=d_dy,
                          integrate_x=ks.integrate_x, integrate_y=ks.integrate_y, integrate_xy=ks.integrate_xy)

    def lift_right(self, ks: KernelSpec, ctx: OpContext) -> KernelSpec:
        ay = self.axis
        if ks.d_dy is not None:
            d_dy_hook = cast(Callable[[Array, Array, int], Array], ks.d_dy)
            k0p = lambda X, Y, h=d_dy_hook: h(X, Y, ay)

            d2_xy_hook = ks.d2_xy
            d2_yy_hook = ks.d2_yy
            d_dx_next: Optional[Callable[[Array, Array, int], Array]] = (
                (lambda X, Y, i, h=cast(Callable[[Array, Array, int, int], Array], d2_xy_hook): h(X, Y, i, ay))
                if d2_xy_hook is not None else None
            )
            d_dy_next: Optional[Callable[[Array, Array, int], Array]] = (
                (lambda X, Y, j, h=cast(Callable[[Array, Array, int, int], Array], d2_yy_hook): h(X, Y, ay, j))
                if d2_yy_hook is not None else None
            )
            return KernelSpec(k0=k0p, d_dx=d_dx_next, d_dy=d_dy_next,
                              d2_xx=None, d2_yy=None, d2_xy=None,
                              integrate_x=ks.integrate_x, integrate_y=ks.integrate_y, integrate_xy=ks.integrate_xy)

        # AD fallback
        k0p = _partial_rows_y(ks.k0, ay)  # ∂/∂y_ay k
        def d_dx(X: Array, Y: Array, i: int) -> Array:
            _, dx = jax.jvp(lambda X_: k0p(X_, Y), (X,), (_dir_tangent_like(X, i),))
            return dx

        return KernelSpec(k0=k0p, d_dx=d_dx,
                          integrate_x=ks.integrate_x, integrate_y=ks.integrate_y, integrate_xy=ks.integrate_xy)


    # pytree plumbing
    def tree_flatten(self):
        return (), (self.axis,)
    
    @classmethod
    def tree_unflatten(cls, aux, children):
        (axis,) = aux
        return cls(axis)
