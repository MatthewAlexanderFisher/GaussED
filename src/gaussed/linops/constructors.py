from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Tuple, Optional, Protocol, Any
import jax
import jax.numpy as jnp
from jax import Array

from gaussed.domains.base import Domain
from gaussed.utils.shape_helpers import _prod, _as_2d, _restore
from gaussed.linops.linop import LinearOp
from gaussed.gp.gp_ops.probe import ProbeStack, _lift_left_all, _lift_right_all
from gaussed.gp.gp_ops.base import KernelSpec, OpContext


# ---- Protocol: a constructor that returns a LinearOp ------------------------
class LinOpConstructor(Protocol):
    def __call__(
        self,
        ks: KernelSpec,
        F: ProbeStack,                   # (nF, *input_shape)
        G: ProbeStack,                   # (nG, *input_shape)
        ctx: OpContext
    ) -> LinearOp: ...


# ---- Dense: materialise K, wrap as LinearOp --------------------------------
@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class DenseGramConstructor:
    def __call__(self, ks: "KernelSpec", F: "ProbeStack", G: "ProbeStack", ctx: "OpContext") -> "LinearOp":
        K = F.kernel(G, ks, ctx)                         # (nF, nG, *L, *R)
        nF, nG = K.shape[:2]
        tail = K.shape[2:]                           # (*L, *R)

        ksL  = _lift_left_all(F.probes[0].ops, ks, ctx)
        ksLR = _lift_right_all(G.probes[0].ops, ksL, ctx)
        L_shape = tuple(getattr(ksLR, "left_shape", ()) or ())
        R_shape = tuple(getattr(ksLR, "right_shape", ()) or ())
        lrank, rrank = len(L_shape), len(R_shape)


        Ls = int(jnp.prod(jnp.array(L_shape)))  if ks.left_shape  else 1
        Rs = int(jnp.prod(jnp.array(R_shape))) if ks.right_shape else 1
        K2 = K.reshape(nF * Ls, nG * Rs)

        def _as_2d(v): return (v[:, None], True) if v.ndim == 1 else (v, False)
        def _restore(y, was_vec): return y.squeeze(-1) if was_vec else y
        def mv(v): V2, s = _as_2d(v);  return _restore(K2 @ V2, s)
        def rmv(w): W2, s = _as_2d(w); return _restore(K2.T @ W2, s)
        return LinearOp((nF * Ls, nG * Rs), mv=mv, rmv=rmv, to_dense=lambda: K2)

    # trivial pytree
    def tree_flatten(self): return (), ()
    @classmethod
    def tree_unflatten(cls, aux, children): return cls()

# ---- MV-only: build mv/rmv with vmaps/einsums (no full K materialisation) ---



def TensorGramOp(k, X: Array, Y: Array, domain, to_dense: bool = False):
    """
    Linear operator for K(X,Y), where k(x,y) has shape (*left,*right).
    Operator shape: (nx*L, ny*R), L=prod(left), R=prod(right).
    mv:  (ny*R,) or (ny*R,k) -> (nx*L,) or (nx*L,k)
    rmv: (nx*L,) or (nx*L,k) -> (ny*R,) or (ny*R,k)
    """
    nx = X.shape[0]; ny = Y.shape[0]
    left  = tuple(getattr(k, "left_shape", ()))
    right = tuple(getattr(k, "right_shape", ()))
    L = _prod(left) if left else 1
    R = _prod(right) if right else 1

    def _as_2d(V: Array):
        return (V[:, None], True) if V.ndim == 1 else (V, False)
    def _restore(U: Array, was_vec: bool):
        return U.squeeze(-1) if was_vec else U

    # v_flat: (ny*R, k)  => out: (nx*L, k)
    def mv(v_flat: Array) -> Array:
        V2, was_vec = _as_2d(v_flat)            # (ny*R, k)
        kR = V2.shape[1]
        Vyrk = V2.reshape(ny, R, kR)            # (ny, R, k)

        def row_apply(x):
            # (ny, *L, *R) -> (ny, L, R)
            KxY = jax.vmap(lambda y: k(x, y, domain))(Y).reshape(ny, L, R)
            # sum over y and r: (ny,L,R) • (ny,R,k) -> (L,k)
            return jnp.einsum("ylr,yrk->lk", KxY, Vyrk)     # (L, k)

        out = jax.vmap(row_apply)(X)             # (nx, L, k)
        out2d = out.reshape(nx * L, kR)          # (nx*L, k)
        return _restore(out2d, was_vec)

    # w_flat: (nx*L, k)  => out: (ny*R, k)
    def rmv(w_flat: Array) -> Array:
        W2, was_vec = _as_2d(w_flat)            # (nx*L, k)
        kR = W2.shape[1]
        Wxlk = W2.reshape(nx, L, kR)            # (nx, L, k)

        def col_apply(y):
            # (nx, *L, *R) -> (nx, L, R)
            KYx = jax.vmap(lambda x: k(x, y, domain))(X).reshape(nx, L, R)
            # sum over x and l: (nx,L,R) • (nx,L,k) -> (R,k)
            return jnp.einsum("xlr,xlk->rk", KYx, Wxlk)     # (R, k)

        out = jax.vmap(col_apply)(Y)            # (ny, R, k)
        out2d = out.reshape(ny * R, kR)         # (ny*R, k)
        return _restore(out2d, was_vec)

    def dense():
        # (nx,ny,*L,*R) -> (nx,ny,L,R) -> (nx,L,ny,R) -> (nx*L,ny*R)
        Kxy = jax.vmap(lambda x: jax.vmap(lambda y: k(x, y, domain))(Y))(X)
        Kxy = Kxy.reshape(nx, ny, L, R)
        return jnp.transpose(Kxy, (0, 2, 1, 3)).reshape(nx * L, ny * R)

    return LinearOp(
        shape=(nx * L, ny * R),
        mv=mv,
        rmv=rmv,
        to_dense=(dense if to_dense else None),
    )

