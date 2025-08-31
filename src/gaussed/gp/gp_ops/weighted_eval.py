from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable, cast
import jax
import jax.numpy as jnp
from jax import Array

from gaussed.gp.gp_ops.base import Functional, FunSpec, KernelSpec, OpContext
from gaussed.domains.base import Domain

def _row_contract(W: Array, A: Array) -> Array:
    """Contract along first axis: W(k,nF) ⨯ A(nF, …) -> (k, …)."""
    return jnp.tensordot(W, A, axes=((1,), (0,)))

def _contract_right_axis_with_W(RX: Array, W: Array) -> Array:
    """
    RX: (k_left, n_right, *T)
    W : (k_right, n_right)
    returns: (k_left, k_right, *T)
    """
    out = jnp.tensordot(RX, W.T, axes=((1,), (0,)))  # (k_left, *T, k_right)
    return jnp.moveaxis(out, -1, 1)                  # (k_left, k_right, *T)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class WeightedEval(Functional):
    """
    Linear functional over evaluations at X:
        y = W @ g(X)
    Shapes:
        X : (nF, d)
        W : (k, nF)
        g(X) : (nF, *L)          -> returns (k, *L)
        ks(X,Y): (nF, nG, *L, *R) -> returns (k, nG, *L, *R)
    """
    W: Array          # (k, nF)
    X: Array          # (nF, d)

    # ----- apply to a function spec: g -> Array (k, *L)
    def __call__(self, g: FunSpec, ctx: OpContext) -> Array:
        W, X = self.W, self.X
        GX = g.eval(X)                                   # (nF, *L)
        return _row_contract(W, GX)                      # (k, *L)

    # ----- left kernel map: ks -> FunSpec over Y with eval(Y)->(k, nG, *L, *R)
    def left_kernel_map(self, ks: KernelSpec, ctx: OpContext) -> FunSpec:
        W, X = self.W, self.X

        def f_eval(Y: Array) -> Array:
            K = ks.__call__(X, Y)                        # (nF, nG, *L, *R)
            return _row_contract(W, K)                   # (k, nG, *L, *R)

        # Optional analytic ∫ over Y
        f_integrate: Optional[Callable[[Domain], Optional[Array]]] = None
        if ks.integrate_y_of is not None:
            iy_of = ks.integrate_y_of
            def _integ(domY: Domain) -> Optional[Array]:
                Fy = iy_of(domY)                         # Optional[Callable[[Array], Array]]
                if Fy is None:
                    return None
                K_int = Fy(X)                            # (nF, *R) or (nF, nR, ...) per your ks
                return _row_contract(W, K_int)           # (k, *R) / (k, nR, ...)
            f_integrate = _integ

        # d/dY
        if ks.d_dy is not None:
            h = cast(Callable[[Array, Array, int], Array], ks.d_dy)
            f_partial = lambda Y, j, h=h, X=X: _row_contract(W, h(X, Y, j))
        else:
            f_partial = None


        # d2/dY^2
        if ks.d2_yy is not None:
            h2 = cast(Callable[[Array, Array, int, int], Array], ks.d2_yy)
            f_partial2 = lambda Y, i, j, h2=h2, X=X: _row_contract(W, h2(X, Y, i, j))
        else:
            f_partial2 = None

        return FunSpec(eval=f_eval, integrate=f_integrate, partial=f_partial, partial2=f_partial2)

    # ----- right reduction: F(Y) expected (nF, …) -> (k, …)
    def right_reduce(self, F: FunSpec, ctx: OpContext) -> Array:
        RX = F.eval(self.X)  # expected (k_left, n_right, *L, *R)
        return _contract_right_axis_with_W(RX, self.W)

    # ----- canonical pair -----
    def pair(self, other: Functional, ks: KernelSpec, ctx: OpContext) -> Array:
        F = self.left_kernel_map(ks, ctx)
        return other.right_reduce(F, ctx)

    # ----- pytree -----
    def tree_flatten(self):
        return (self.W, self.X), ()
    @classmethod
    def tree_unflatten(cls, aux, ch):
        W, X = ch
        return cls(W, X)
