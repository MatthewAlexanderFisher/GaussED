from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Protocol, Optional, Tuple, Callable, cast
from jax import Array
import jax.numpy as jnp
import jax

from gaussed.gp.gp_ops.base import Operator, Functional, OpContext, FunSpec, KernelSpec, ShapeNeutralOp
from gaussed.domains.base import Domain
from gaussed.codomains.base import Codomain
from gaussed.utils.shape_helpers import _prod, _append, _append_flat, _append_full

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Grad:
    """Gradient/Jacobian operator using JAX shape: (n, *out, *in).
       - use_rev: choose jacrev (True) vs jacfwd (False)
       - flatten_input_axes: append (prod(in),) instead of *in (set True if you want 1D grad axis)
    """
    use_rev: bool = False
    flatten_input_axes: bool = False

    # ---------- shape transforms (USED below) ----------
    def output_codomain(self, cod: Codomain, dom: Domain) -> Codomain:
        in_shape = dom.input_shape
        new_tail = ( _prod(in_shape), ) if self.flatten_input_axes else in_shape
        return Codomain(_append_full(cod.output_shape, new_tail))

    def map_left_shape(self, left_shape: Tuple[int, ...], dom: Domain) -> Tuple[int, ...]:
        in_shape = dom.input_shape
        return _append_flat(left_shape, in_shape) if self.flatten_input_axes \
               else _append_full(left_shape, in_shape)

    def map_right_shape(self, right_shape: Tuple[int, ...], dom: Domain) -> Tuple[int, ...]:
        in_shape = dom.input_shape
        return _append_flat(right_shape, in_shape) if self.flatten_input_axes \
               else _append_full(right_shape, in_shape)

    # ---------- unary on means: FunSpec -> FunSpec ----------
    def __call__(self, g: FunSpec, ctx: OpContext) -> FunSpec:
        dom = ctx.domain
        out_cod = self.output_codomain(g.codomain, dom)

        # single point: (*in,) -> (*out,)
        def f_single(x: Array) -> Array:
            return g.eval(x[None, ...])[0]

        jac = jax.jacrev(f_single) if self.use_rev else jax.jacfwd(f_single)

        def eval_grad(X: Array) -> Array:
            Xn = dom.ensure_inputs(X)                     # (n,*in)
            J  = jax.vmap(jac)(Xn)                        # (n,*out,*in)   [JAX layout]
            # if flatten_input_axes=True and in_shape has >1 dims, jac already returns (*out,*in),
            # but out_cod encodes the flattened tail; reshape to match
            want = out_cod.output_shape                   # (*out, tail)
            if J.shape[1:] != want:
                n = J.shape[0]
                return J.reshape((n, *want))
            return J

        return FunSpec(eval=eval_grad, codomain=out_cod,
                       integrate=None, partial=None, partial2=None)

    # ---------- kernel lifts ----------
    def lift_left(self, ks: KernelSpec, ctx: OpContext) -> KernelSpec:
        """k(X,Y) -> ∇_X k(X,Y); result (nX, nY, *newL, *R)."""
        dom   = ks.domain
        in_shape = dom.input_shape
        d = _prod(in_shape) if self.flatten_input_axes else len(in_shape)
        new_L = self.map_left_shape(ks.left_shape, dom)  # use the shape-map

        def k0p(X: jnp.ndarray, Y: jnp.ndarray) -> jnp.ndarray:
            Xn = dom.ensure_inputs(X)            # (nX,*in)
            Yn = dom.ensure_inputs(Y)            # (nY,*in)

            # g_x: (*in,) -> (nY, *L, *R)
            def g_x(x: jnp.ndarray) -> jnp.ndarray:
                K = ks(x[None, ...], Yn)         # (1, nY, *L, *R)
                return K[0]                      # (nY, *L, *R)

            jac = jax.jacrev(g_x) if self.use_rev else jax.jacfwd(g_x)
            J   = jax.vmap(jac)(Xn)              # (nX, nY, *L, *R, *in)

            # Option A: flatten input axes to 'd' and move it just after *L
            if self.flatten_input_axes and in_shape:
                Llen = len(ks.left_shape)
                Rlen = len(ks.right_shape)
                J = J.reshape(
                    (Xn.shape[0], Yn.shape[0], *ks.left_shape, *ks.right_shape, d)
                )                                 # (nX, nY, *L, *R, d)
                # move the last axis 'd' to position 2+Llen  → (nX, nY, *L, d, *R)
                J = jnp.moveaxis(J, -1, 2 + Llen)
                return J

            # Option B: keep *in; move the whole block *in before *R
            if in_shape:
                Llen = len(ks.left_shape)
                Rlen = len(ks.right_shape)
                axes = list(range(J.ndim))
                # indices of the input-derivative block at the tail
                blk = list(range(2 + Llen + Rlen, 2 + Llen + Rlen + len(in_shape)))
                # remove block then insert before the right block
                rest = [a for a in axes if a not in blk]
                insert_at = 2 + Llen
                new_axes = rest[:insert_at] + blk + rest[insert_at:]
                J = jnp.transpose(J, new_axes)    # (nX, nY, *L, *in, *R)
            return J

        # update ks.codomain 
        new_cod = Codomain((*new_L, *ks.right_shape))
        return KernelSpec(ks.domain, new_cod, k0p, new_L, ks.right_shape)

    def lift_right(self, ks: KernelSpec, ctx: OpContext) -> KernelSpec:
        """k(X,Y) -> ∇_Y k(X,Y); result (nX, nY, *L, *newR)."""
        dom   = ks.domain
        new_R = self.map_right_shape(ks.right_shape, dom)

        def k0p(X: Array, Y: Array) -> Array:
            Xn = dom.ensure_inputs(X)                    # (nX,*in)
            Yn = dom.ensure_inputs(Y)                    # (nY,*in)

            # g_y: (*in,) -> (nX, *L, *R)
            def g_y(y: Array) -> Array:
                K = ks(Xn, y[None, ...])                 # (nX, 1, *L, *R)
                return K[:, 0, ...]                      # (nX, *L, *R)

            jac = jax.jacrev(g_y) if self.use_rev else jax.jacfwd(g_y)
            J_each = jax.vmap(jac)(Yn)                   # (nY, nX, *L, *R, *in)
            J = jnp.swapaxes(J_each, 0, 1)               # (nX, nY, *L, *R, *in)
            if self.flatten_input_axes and dom.input_shape != ():
                tail = (*ks.left_shape, *ks.right_shape, _prod(dom.input_shape))
                return J.reshape((Xn.shape[0], Yn.shape[0], *tail))
            return J

        new_cod = getattr(ks, "codomain", None)
        if new_cod is not None:
            new_cod = Codomain((*ks.left_shape, *new_R))
        return replace(ks, k0=k0p, right_shape=new_R, codomain=new_cod) if new_cod is not None \
               else replace(ks, k0=k0p, right_shape=new_R)

    # pytree
    def tree_flatten(self): return (), (self.use_rev, self.flatten_input_axes)
    @classmethod
    def tree_unflatten(cls, aux, ch):
        use_rev, flatten = aux
        return cls(use_rev, flatten)
