from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol, Optional, Tuple, Callable, cast
from jax import Array
import jax.numpy as jnp
import jax

from gaussed.gp.gp_ops.base import Operator, Functional, OpContext, FunSpec, KernelSpec
from gaussed.utils.shape_helpers import _prod, _ensure_2d_rows
from gaussed.domains.base import Domain
from gaussed.codomains.base import Codomain


# --- Affine operator: f ↦ A f + c --------------------------------------------
@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Affine:
    A: Array                      # (P, Q)
    c: Optional[Array] = None     # (P,) or broadcastable to (P,)

    enable_checks: bool = field(default=True, repr=False)

    # ===== Unary lift on functions: FunSpec -> FunSpec ========================
    def __call__(self, g: "FunSpec", ctx: "OpContext") -> "FunSpec":
        A = self.A
        c = self.c
        P, Q = int(A.shape[0]), int(A.shape[1])

        # base eval(X): (n, q)  ->  (n, P)
        def ev(X: Array) -> Array:
            G = g.eval(X)                     # (n, q) or (n, *out)
            G2 = _ensure_2d_rows(G)           # (n, q)
            if self.enable_checks:
                assert G2.shape[1] == Q, f"[Affine] A expects Q={Q} but f has {G2.shape[1]}"
            H = G2 @ A.T                      # (n, P)
            if c is not None:
                H = H + jnp.asarray(c).reshape((1, P))
            return H

        # propagate analytic partials if present (linear op -> easy)
        p1 = g.partial
        if p1 is not None:
            p1 = cast(Callable[[Array, int], Array], p1)
            def nxt_partial(X: Array, j: int) -> Array:
                G = p1(X, j)                  # (n, q)
                return _ensure_2d_rows(G) @ A.T
            p2_src = g.partial2
            if p2_src is not None:
                p2_src = cast(Callable[[Array, int, int], Array], p2_src)
                def nxt_partial2(X: Array, i: int, j: int) -> Array:
                    G = p2_src(X, i, j)
                    return _ensure_2d_rows(G) @ A.T
                return FunSpec(eval=ev, partial=nxt_partial, partial2=nxt_partial2)
            return FunSpec(eval=ev, partial=nxt_partial, partial2=None)

        return FunSpec(eval=ev, partial=None, partial2=None)

    # ===== Kernel lifts (left/right) ==========================================
    def lift_left(self, ks: "KernelSpec", ctx: "OpContext") -> "KernelSpec":
        A = self.A
        P, Q = int(A.shape[0]), int(A.shape[1])
        L, R = ks.left_shape, ks.right_shape
        Ls, Rs = _prod(L), _prod(R)
        if self.enable_checks:
            assert Ls == Q, f"[Affine.lift_left] left size {Ls} != A second dim {Q}"

        def _left_mul(K: Array) -> Array:
            # K: (n_x, n_y, *L, *R) -> (n_x, n_y, P, *R)
            n_x, n_y = K.shape[:2]
            K2 = K.reshape(n_x, n_y, Ls, Rs)                 # (nx,ny,Q,Rs)
            out = jnp.einsum("pq,fgqk->fgpk", A, K2)         # left-multiply by A
            return out.reshape(n_x, n_y, P, *R)

        # base kernel
        def k0p(X: Array, Y: Array) -> Array:
            return _left_mul(ks(X, Y))

        # wrap derivative hooks linearly if provided
        d_dx = ks.d_dx
        d_dy = ks.d_dy
        d2_xx = ks.d2_xx
        d2_yy = ks.d2_yy
        d2_xy = ks.d2_xy

        def wrap1(h):
            return (lambda X, Y, i, h=h: _left_mul(h(X, Y, i))) if h is not None else None
        def wrap2(h):
            return (lambda X, Y, i, j, h=h: _left_mul(h(X, Y, i, j))) if h is not None else None

        return KernelSpec(
            domain=ks.domain, codomain=ks.codomain,
            k0=k0p,
            left_shape=(P,), right_shape=R,
            d_dx=wrap1(d_dx), d_dy=wrap1(d_dy),
            d2_xx=wrap2(d2_xx), d2_yy=wrap2(d2_yy), d2_xy=wrap2(d2_xy),
            integrate_x_of=None, integrate_y_of=None, integrate_xy_of=None
        )

    def lift_right(self, ks: "KernelSpec", ctx: "OpContext") -> "KernelSpec":
        A = self.A
        P, Q = int(A.shape[0]), int(A.shape[1])
        L, R = ks.left_shape, ks.right_shape
        Ls, Rs = _prod(L), _prod(R)
        if self.enable_checks:
            assert Rs == Q, f"[Affine.lift_right] right size {Rs} != A second dim {Q}"

        AT = A.T  # (Q, P)

        def _right_mul(K: Array) -> Array:
            # K: (n_x, n_y, *L, *R) -> (n_x, n_y, *L, P)
            n_x, n_y = K.shape[:2]
            K2 = K.reshape(n_x, n_y, Ls, Rs)                 # (nx,ny,Ls,Q)
            out = jnp.einsum("fgik,kp->fgip", K2, AT)        # right-multiply by A^T
            return out.reshape(n_x, n_y, *L, P)

        def k0p(X: Array, Y: Array) -> Array:
            return _right_mul(ks(X, Y))

        # wrap derivative hooks
        d_dx = ks.d_dx
        d_dy = ks.d_dy
        d2_xx = ks.d2_xx
        d2_yy = ks.d2_yy
        d2_xy = ks.d2_xy

        def wrap1(h):
            return (lambda X, Y, i, h=h: _right_mul(h(X, Y, i))) if h is not None else None
        def wrap2(h):
            return (lambda X, Y, i, j, h=h: _right_mul(h(X, Y, i, j))) if h is not None else None

        return KernelSpec(
            domain=ks.domain, codomain=ks.codomain,
            k0=k0p,
            left_shape=L, right_shape=(P,),
            d_dx=wrap1(d_dx), d_dy=wrap1(d_dy),
            d2_xx=wrap2(d2_xx), d2_yy=wrap2(d2_yy), d2_xy=wrap2(d2_xy),
            integrate_x_of=None, integrate_y_of=None, integrate_xy_of=None
        )

    # TODO: implement
    def output_codomain(self, cod: "Codomain", dom: "Domain") -> "Codomain": ...
    def map_left_shape(self, left_shape: Tuple[int, ...], dom: "Domain") -> Tuple[int, ...]: ...
    def map_right_shape(self, right_shape: Tuple[int, ...], dom: "Domain") -> Tuple[int, ...]: ...


    # pytree
    def tree_flatten(self): return (), (self.A, self.c)
    @classmethod
    def tree_unflatten(cls, aux, ch):
        A, c = aux
        return cls(A, c)
