from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol, Optional, Tuple, Callable, TYPE_CHECKING
from jax import Array
import jax

if TYPE_CHECKING:
    from gaussed.backends.solvers.quadrature import Quadrature, BiQuadrature
    from gaussed.domains.base import Domain

from gaussed.utils.shape_helpers import _enforce_event_shape, _ensure_n_by_d

# === Function and Kernel Specs ===============================================

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class FunSpec:
    eval: Callable[[Array], Array]                 # (n,*in_shape) -> (n,*out_shape)
    out_shape: Tuple[int, ...] = field(default_factory=lambda: (1,))

    # Integral hook:
    # Try an analytic integral over a requested Y-domain. Return None if unsupported.
    integrate: Optional[Callable[[Domain], Optional[Array]]] = None
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
    # Base geometry domain (distance/geometry lives here fixed to kernel)
    domain: Domain

    # Base kernel already bound to 'domain'
    k0: Callable[[Array, Array], Array]

    # Tensor shapes
    left_shape: Tuple[int, ...]
    right_shape: Tuple[int, ...]
    
    # Integration hooks are FACTORIES that accept the integration domain(s)
    # If None -> caller should fall back to quadrature.
    integrate_x_of: Optional[Callable[[Domain], Optional[Callable[[Array], Array]]]] = None  # domX -> maybe(Y->∫_X k)
    integrate_y_of: Optional[Callable[[Domain], Optional[Callable[[Array], Array]]]] = None  # domY -> maybe(X->∫_Y k)
    integrate_xy_of: Optional[Callable[[Domain, Domain], Optional[Array]]] = None  # maybe ∬ k

    # Derivative hooks remain 2-arg and already bound to base 'domain'
    d_dx:  Optional[Callable[[Array, Array, int], Array]] = None
    d_dy:  Optional[Callable[[Array, Array, int], Array]] = None
    d2_xx: Optional[Callable[[Array, Array, int, int], Array]] = None
    d2_yy: Optional[Callable[[Array, Array, int, int], Array]] = None
    d2_xy: Optional[Callable[[Array, Array, int, int], Array]] = None

    def __call__(self, X: Array, Y: Array) -> Array:
        X = _ensure_n_by_d(X)
        Y = _ensure_n_by_d(Y)
        K = self.k0(X, Y)  # may return many shapes depending on implementation
        return _enforce_event_shape(K, X.shape[0], Y.shape[0], self.left_shape, self.right_shape)

    def tree_flatten(self):
        children = (self.domain,)
        aux = (self.k0, self.left_shape, self.right_shape,
               self.integrate_x_of, self.integrate_y_of, self.integrate_xy_of,
               self.d_dx, self.d_dy, self.d2_xx, self.d2_yy, self.d2_xy)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (domain,) = children
        (k0, lshape, rshape,
         ix_of, iy_of, ixy_of,
         d_dx, d_dy, d2_xx, d2_yy, d2_xy) = aux
        return cls(domain, k0, lshape, rshape, ix_of, iy_of, ixy_of,
                   d_dx, d_dy, d2_xx, d2_yy, d2_xy)



# === OpContext to pass info to operators ===========================================

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class OpContext:
    domain: Domain # used in quadrature methods / integral methods
    # Defaults for numeric fallbacks
    quad: Optional["Quadrature"]   = None   # general unary/default
    quad_x: Optional["Quadrature"] = None   # integrate over X (left maps)
    quad_y: Optional["Quadrature"] = None   # integrate over Y (right reductions)
    quad_xy: Optional["BiQuadrature"] = None  # specialised double integral for kernels


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
    def left_kernel_map(self, ks: KernelSpec, ctx: OpContext) -> FunSpec: ...

    # Consumes a FunSpec (function of Y) and realises it under this functional
    def right_reduce(self, F: FunSpec, ctx: OpContext) -> Array: ...

    # default pair implementation (needs to be copied to all classes following this protocol)
    def pair(self, other: "Functional", ks: KernelSpec, ctx: OpContext) -> Array: ...


