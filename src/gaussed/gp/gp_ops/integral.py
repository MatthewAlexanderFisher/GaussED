from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Optional, Tuple, Callable, Any
from jax import Array
import jax.numpy as jnp
import jax

from gaussed.gp.gp_ops.base import FunSpec, KernelSpec, OpContext, Functional

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Integral:
    nodes: Optional[Array] = None            # (q, d), optional if using quad
    weights: Optional[Array] = None          # (n_int, q)
    # optional custom quadrature: takes a function of Y and meta, returns (n_int, n_L) or (n_L, n_int) – we'll define orientation below
    quad_rule: Optional[Callable[[Callable[[Array], Array], Any], Array]] = None
    meta: Any = None

    # unary: integrate g over Y (requires nodes/weights or quad_rule)
    def __call__(self, g: FunSpec, ctx: OpContext) -> Array:
        if self.nodes is not None and self.weights is not None:
            G = g.eval(self.nodes)                  # (q, m)
            return self.weights @ G                 # (n_int, m)
        if self.quad_rule is not None:
            return self.quad_rule(g.eval, self.meta)  # (n_int, m) per your rule
        raise ValueError("Integral needs nodes/weights or quad_rule")

    # left kernel map: Y ↦ k0(X, Y) will be provided by the left functional; Integral doesn't need to override this.
    # We keep the default behavior from the protocol (no override).
    def left_kernel_map(self, ks: KernelSpec, ctx: OpContext) -> Callable[[Array], Array]:
        # Prefer analytic kernel hook if available
        if ks.integrate_x is not None:
            # Convention: kernel decides how to interpret meta for X-side integration
            return lambda Y: ks.integrate_x(Y, self.meta)  # (n_L, q_Y)

        # Else do discrete sum via nodes/weights
        if self.nodes is not None and self.weights is not None:
            nodes, W = self.nodes, self.weights  # nodes: (q_X,d), W: (n_L, q_X)
            return lambda Y: W @ ks.k0(nodes, Y)  # (n_L, q_Y)

        # A generic quad_rule that integrates over Y cannot supply a left map (which integrates over X)
        raise ValueError(
            "Integral.left_kernel_map: need (nodes,weights) or ks.integrate_x "
            "to integrate over X. A Y-side quad_rule is for right_reduce only."
        )

    # right reduction: reduce F(Y) along Y using nodes/weights or quad
    def right_reduce(self, F_of_Y: Callable[[Array], Array], ctx: OpContext) -> Array:
        # Nodes/weights path: F(nodes) is (n_L, q) -> (n_L, n_int)
        if self.nodes is not None and self.weights is not None:
            F_nodes = F_of_Y(self.nodes)            # (n_L, q)
            return F_nodes @ self.weights.T         # (n_L, n_int)

        # Quad path: define a Y-function that returns (q, n_L) so a row-wise quadrature can be applied.
        if self.quad_rule is not None:
            def gY(Y: Array) -> Array:              # (q, d) -> (q, n_L)
                F = F_of_Y(Y)                       # (n_L, q)
                return jnp.swapaxes(F, 0, 1)        # (q, n_L)
            Q = self.quad_rule(gY, self.meta)       # (n_int, n_L) by convention
            return jnp.swapaxes(Q, 0, 1)            # (n_L, n_int)

        raise ValueError("Integral.right_reduce needs nodes/weights or quad_rule")

    def pair(self, other: Functional, ks: KernelSpec, ctx: OpContext) -> Array:
        return other.right_reduce(self.left_kernel_map(ks, ctx), ctx)

    def tree_flatten(self): return ((self.nodes, self.weights, self.meta), (self.quad_rule,))
    @classmethod
    def tree_unflatten(cls, aux, ch):
        nodes, W, meta = ch
        (qr,) = aux
        return cls(nodes, W, qr, meta)
