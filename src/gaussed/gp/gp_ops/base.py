from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Optional, Tuple, Callable, Literal, Any
from jax import Array
import jax

from gaussed.backends.solvers.quadrature import Quadrature, DiscreteQuadrature, BiQuadrature
from gaussed.domains.base import Domain

# === Function and Kernel Specs ===============================================

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class FunSpec:
    eval: Callable[[Array], Array]  # (n,d)->(n,m)

    # Integral hooks: analytic integral if available (depends on domain)
    integrate: Optional[Callable[[Domain], Array]] = None
    # Derivative hooks
    partial: Optional[Callable[[Array, int], Array]] = None
    partial2: Optional[Callable[[Array, int, int], Array]] = None
    def tree_flatten(self): return (), (self.eval, self.integrate, self.partial, self.partial2)
    @classmethod
    def tree_unflatten(cls, aux, ch):
        ev, integ, d1, d2 = aux
        return cls(ev, integ, d1, d2)

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class KernelSpec:
    k0: Callable[[Array, Array], Array]  # (n_x,d),(n_y,d)->(n_x,n_y)

    # Integral hooks: analytic integral if available (depends on domain)
    # Left: Y ↦ ∫_X k0(X,Y) dμ_X
    integrate_x: Optional[Callable[[Domain], Callable[[Array], Array]]] = None  # -> (n_L, q_y)
    # Right: X ↦ ∫_Y k0(X,Y) dμ_Y
    integrate_y: Optional[Callable[[Domain], Callable[[Array], Array]]] = None  # -> (n_x, n_R)
    # Double: ∬ k0(X,Y) dμ_X dμ_Y
    integrate_xy: Optional[Callable[[Domain, Domain], Array]] = None           # -> (n_L, n_R)

    # Derivative hooks (optional)
    d_dx:  Optional[Callable[[Array, Array, int], Array]] = None
    d_dy:  Optional[Callable[[Array, Array, int], Array]] = None
    d2_xx: Optional[Callable[[Array, Array, int, int], Array]] = None
    d2_yy: Optional[Callable[[Array, Array, int, int], Array]] = None
    d2_xy: Optional[Callable[[Array, Array, int, int], Array]] = None

    def tree_flatten(self):
        return (), (
            self.k0, self.integrate_x, self.integrate_y, self.integrate_xy,
            self.d_dx, self.d_dy, self.d2_xx, self.d2_yy, self.d2_xy,
        )
    @classmethod
    def tree_unflatten(cls, aux, ch):
        k0, ix, iy, ixy, dx, dy, dxx, dyy, dxy = aux
        return cls(k0, ix, iy, ixy, dx, dy, dxx, dyy, dxy)


# === OpContext to pass info to operators ===========================================

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class OpContext:
    domain: Domain
    # Defaults for numeric fallbacks
    quad: Optional[Quadrature]   = None   # general unary/default
    quad_x: Optional[Quadrature] = None   # integrate over X (left maps)
    quad_y: Optional[Quadrature] = None   # integrate over Y (right reductions)
    quad_xy: Optional[BiQuadrature] = None  # specialised double integral for kernels


    # pytree: treat callables as static aux
    def tree_flatten(self):
        # treat everything as children so array params trace cleanly
        return ((self.domain, self.quad, self.quad_x, self.quad_y, self.quad_xy), ())
    @classmethod
    def tree_unflatten(cls, aux, ch):
        dom, q, qx, qy, qxy = ch
        return cls(dom, q, qx, qy, qxy)


# === Operator + Functional protocol ===================================

class Operator(Protocol):
    def __call__(self, g: FunSpec) -> FunSpec: ...
    def lift_left(self, ks: KernelSpec, ctx: OpContext) -> KernelSpec: ...
    def lift_right(self, ks: KernelSpec, ctx: OpContext) -> KernelSpec: ...


class Functional(Protocol):
    # unary realisation: FunSpec -> Array
    def __call__(self, g: FunSpec, ctx: OpContext) -> Array: ...

    # build a function of Y that returns the left kernel block (n_L, q)
    def left_kernel_map(self, ks: KernelSpec, ctx: OpContext) -> Callable[[Array], Array]: ...

    # given F(Y) (n_L, q), reduce along Y to produce (n_L, n_R) using this functional’s measure
    def right_reduce(self, F_of_Y: Callable[[Array], Array], ctx: OpContext) -> Array: ...

    # default pair implementation (can be mixed in)
    def pair(self, other: "Functional", ks: KernelSpec, ctx: OpContext) -> Array:
        return other.right_reduce(self.left_kernel_map(ks, ctx), ctx)


# === Probe is a symbolic chain of Operators with a reducer ===============
@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Probe:
    ops: Tuple[Operator, ...]
    fnl: Functional

    # unary
    def apply(self, g: FunSpec, ctx: OpContext) -> Array:
        for op in self.ops: g = op(g)
        return self.fnl(g, ctx)

    # binary (kernel)
    def kernel(self, other: "Probe", ks: KernelSpec, ctx: OpContext) -> Array:
        # left lifts
        ksL = ks
        for op in self.ops:
            ksL = op.lift_left(ksL, ctx)
        # right lifts
        ksLR = ksL
        for op in other.ops:
            ksLR = op.lift_right(ksLR, ctx)
        # realise with the two functionals
        return self.fnl.pair(other.fnl, ksLR, ctx)

    def tree_flatten(self): return ((self.ops, self.fnl), ())
    @classmethod
    def tree_unflatten(cls, aux, ch): ops, fnl = ch; return cls(ops, fnl)
