from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Optional, Tuple, Callable, Any, cast
from jax import Array
import jax.numpy as jnp
import jax

from gaussed.gp.gp_ops.base import FunSpec, KernelSpec, OpContext, Functional
from gaussed.backends.solvers.quadrature import Quadrature
from gaussed.domains.base import Domain 

from typing import Optional, Callable

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Integral:
    # Integrate w.r.t. this (sub)domain; if None, use ctx.domain
    dom: Optional[Domain] = None
    # Optional numeric rule on this side (over X in left map; over Y in right reduce)
    quad: Optional[Quadrature] = None

    # ---------- Unary: ∫ g over dom ----------
    def __call__(self, g: FunSpec, ctx: OpContext) -> Array:
        dom = self.dom or ctx.domain
        quad = self.quad or ctx.quad or ctx.quad_y
        return _integrate_unary_or_quad(g, dom, quad)

    # ---------- Left kernel map: Y ↦ ∫_X k(X,Y) dμ_X ----------
    def left_kernel_map(self, ks: KernelSpec, ctx: OpContext) -> FunSpec:
        domX = self.dom or ctx.domain
        qx = self.quad or ctx.quad_x or ctx.quad  # Optional[Quadrature]

        # 1) Build f_eval(Y) = ∫_X k(X,Y) dμ_X over domX
        f_eval: Callable[[Array], Array] = _pick_integrate_x(ks, domX, qx)

        # 2) Optional double-integral ∬ k dμ_X dμ_Y as a *maybe-analytic* (domY)->Optional[Array]
        f_integrate: Optional[Callable[[Domain], Optional[Array]]] = None
        if ks.integrate_xy_of is not None:
            ixy_of = ks.integrate_xy_of  # Callable[[Domain, Domain], Optional[Array]]
            def _integ(domY: Domain, ixy_of=ixy_of, domX=domX) -> Optional[Array]:
                return ixy_of(domX, domY)
            f_integrate = _integ  # right_reduce will fall back if this returns None

        # 3) Derivative evals via numeric-in-X if hooks exist and qx present
        f_partial: Optional[Callable[[Array, int], Array]] = None
        if ks.d_dy is not None and qx is not None:
            d_dy = ks.d_dy
            f_partial = lambda Y, j, d_dy=d_dy, qx=qx, domX=domX: qx.apply(lambda X: d_dy(X, Y, j), domX)

        f_partial2: Optional[Callable[[Array, int, int], Array]] = None
        if ks.d2_yy is not None and qx is not None:
            d2_yy = ks.d2_yy
            f_partial2 = lambda Y, i, j, d2_yy=d2_yy, qx=qx, domX=domX: qx.apply(lambda X: d2_yy(X, Y, i, j), domX)

        # FunSpec.integrate should have type Optional[Callable[[Domain], Optional[Array]]]
        return FunSpec(eval=f_eval, integrate=f_integrate, partial=f_partial, partial2=f_partial2)

    # ---------- Right reduction: integrate a FunSpec(Y) over Y ----------
    def right_reduce(self, F: FunSpec, ctx: OpContext) -> Array:
        domY = self.dom or ctx.domain
        # Try analytic integral over domY if provided and supported
        if F.integrate is not None:
            out = F.integrate(domY)
            if out is not None:
                return out
        # Numeric fallback
        quadY = self.quad or ctx.quad_y or ctx.quad
        if quadY is None:
            raise ValueError("Integral.right_reduce: need quadrature (self.quad or ctx.quad/quad_y).")
        def gY(Y: Array) -> Array:       # (q_y,d)->(q_y, n_L)
            V = F.eval(Y)                # (n_L, q_y)
            return jnp.swapaxes(V, 0, 1) # (q_y, n_L)
        Q = quadY.apply(gY, domY)        # (n_R, n_L)
        return jnp.swapaxes(Q, 0, 1)     # (n_L, n_R)

    # ---------- Pair: fast paths then fallback ----------
    def pair(self, other: Functional, ks: KernelSpec, ctx: OpContext) -> Array:
        if isinstance(other, Integral):
            domX = self.dom or ctx.domain
            domY = other.dom or ctx.domain

            # 1) Analytic ∬ if available for these specific subdomains
            if ks.integrate_xy_of is not None:
                val = ks.integrate_xy_of(domX, domY)
                if val is not None:
                    return val

            # 2) Dedicated bi-quadrature
            if ctx.quad_xy is not None:
                return ctx.quad_xy.apply2(ks.k0, domX, domY)

        # 3) Fallback composition
        F = self.left_kernel_map(ks, ctx)
        return other.right_reduce(F, ctx)

    # pytree plumbing
    def tree_flatten(self):
        return ((self.dom, self.quad), ())

    @classmethod
    def tree_unflatten(cls, _aux, ch):
        dom, quad = ch
        return cls(dom, quad)


# -------------- helpers --------------

def _integrate_unary_or_quad(g: FunSpec, dom: Domain, quad: Optional[Quadrature]) -> Array:
    """Compute ∫ g over dom, trying analytic first; fall back to quadrature."""
    if g.integrate is not None:
        out = g.integrate(dom)  # Optional[Array]
        if out is not None:
            return out
    if quad is None:
        raise ValueError("Integral: need quadrature to integrate unary FunSpec when no analytic form is available.")
    return quad.apply(g.eval, dom)

def _pick_integrate_x(ks: KernelSpec, domX: Domain, qx: Optional[Quadrature]) -> Callable[[Array], Array]:
    """Return Y -> ∫_X k(X,Y) over domX, using analytic factory if it exists for domX; else quadrature."""
    if ks.integrate_x_of is not None:
        f = ks.integrate_x_of(domX)  # Optional[Callable[[Array], Array]]
        if f is not None:
            return f
    if qx is None:
        raise ValueError("Integral.left_kernel_map: need quad_x (or self.quad/ctx.quad) when no analytic integrate_x.")
    k0 = ks.k0
    return lambda Y, qx=qx, domX=domX, k0=k0: qx.apply(lambda X: k0(X, Y), domX)
