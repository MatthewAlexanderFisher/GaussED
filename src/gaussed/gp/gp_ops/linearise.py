from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Tuple, Optional
import jax
import jax.numpy as jnp
from jax import Array

from gaussed.gp.gp_ops.base import Operator, Functional, OpContext, FunSpec, KernelSpec
from gaussed.types import ProbeLike
from gaussed.utils.shape_helpers import _prod

def _ensure_event(u: Array) -> Array:
    # promote (...,) -> (...,1) to keep a proper event axis
    return u[..., None] if u.ndim == 1 else u


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Linearise:
    forward: Callable[[Array], Array]  # (..., P) -> (..., P)
    jacobian: Optional[Callable[[Array], Array] ]
    ref_fn: Callable[[Array], Array] 

    def _jac(self) -> Callable[[Array], Array] :
        if self.jacobian is not None:
            return self.jacobian

        fwd = self.forward

        def single_jac(v: Array) -> Array:              # v: (P,)
            def g1(z):                                  # z: (P,) -> (P,)
                y = fwd(z[jnp.newaxis, ...])            # (1,P)
                return y[0]
            return jax.jacfwd(g1)(v)                    # (P,P)

        def jac(u: Array) -> Array:                     # u: (...,P)
            u = _ensure_event(u)
            P = u.shape[-1]
            U = u.reshape((-1, P))                      # (N,P)
            J = jax.vmap(single_jac)(U)                 # (N,P,P)
            return J.reshape(*u.shape[:-1], P, P)       # (...,P,P)
        return jac


    # ---------- FunSpec lift ----------
    def __call__(self, g: "FunSpec", ctx: "OpContext") -> "FunSpec":
        fwd, ref, jac = self.forward, self.ref_fn, self._jac()

        def ev(X: Array) -> Array:
            f  = _ensure_event(g.eval(X))               # (...,P)
            f0 = _ensure_event(ref(X))                  # (...,P)
            A  = jac(f0)                                # (...,P,P)
            # b = g(f0) - A @ f0
            Af0 = jnp.matmul(A, f0[..., None])[..., 0]  # (...,P)
            b   = _ensure_event(fwd(f0)) - Af0          # (...,P)
            Af  = jnp.matmul(A, f[...,   None])[..., 0] # (...,P)
            return b + Af                                # (...,P)

        return FunSpec(eval=ev, partial=None, partial2=None)

    # ---------- Kernel lifts ----------
    def lift_left(self, ks: "KernelSpec", ctx: "OpContext") -> "KernelSpec":
        ref, jac = self.ref_fn, self._jac()
        cod = ks.codomain

        def k0p(X: Array, Y: Array) -> Array:
            K  = ks.__call__(X, Y)                           # (nx,ny,*L,*R)
            nx, ny = X.shape[0], Y.shape[0]
            Ls = _prod(ks.left_shape); Rs = _prod(ks.right_shape)
            K2 = K.reshape(nx, ny, Ls, Rs)                   # (nx,ny,P,R)  P=Ls
            A  = jac(_ensure_event(ref(X)))                  # (nx,P,P)
            Aexp = A[:, None, ...]                           # (nx,1,P,P) to broadcast over ny
            out = jnp.einsum("...qp,...pr->...qr", Aexp, K2) # (nx,ny,P,R)
            return out.reshape(*K.shape[:-2], *K.shape[-2:])

        return KernelSpec(domain=ks.domain, codomain=cod, k0=k0p,
                          left_shape=ks.left_shape, right_shape=ks.right_shape,
                          d_dx=None, d_dy=None, d2_xx=None, d2_yy=None, d2_xy=None,
                          integrate_x_of=None, integrate_y_of=None, integrate_xy_of=None)

    def lift_right(self, ks: "KernelSpec", ctx: "OpContext") -> "KernelSpec":
        ref, jac = self.ref_fn, self._jac()
        cod = ks.codomain

        def k0p(X: Array, Y: Array) -> Array:
            K  = ks.__call__(X, Y)                           # (nx,ny,*L,*R)
            nx, ny = X.shape[0], Y.shape[0]
            Ls = _prod(ks.left_shape); Rs = _prod(ks.right_shape)
            K2 = K.reshape(nx, ny, Ls, Rs)                   # (nx,ny,L,P)  P=Rs
            A  = jac(_ensure_event(ref(Y)))                  # (ny,P,P)
            AT = jnp.swapaxes(A, -1, -2)                     # (ny,P,P)
            ATexp = AT[None, ...]                            # (1,ny,P,P) to broadcast over nx
            out = jnp.einsum("...pr,...rq->...pq", K2, ATexp)# (nx,ny,L,P)
            return out.reshape(*K.shape[:-2], *K.shape[-2:])

        return KernelSpec(domain=ks.domain, codomain=cod, k0=k0p,
                          left_shape=ks.left_shape, right_shape=ks.right_shape,
                          d_dx=None, d_dy=None, d2_xx=None, d2_yy=None, d2_xy=None,
                          integrate_x_of=None, integrate_y_of=None, integrate_xy_of=None)

    # ---- pytree plumbing (treat callables as static aux) ----
    def tree_flatten(self):
        children = ()  # no array children
        aux = (self.forward, self.jacobian, self.ref_fn)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        forward, jacobian, ref_fn = aux
        return cls(forward, jacobian, ref_fn)
