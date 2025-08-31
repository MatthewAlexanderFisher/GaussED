from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Tuple, Optional, Protocol, Any
import jax
import jax.numpy as jnp
from jax import Array
import math

from gaussed.domains.base import Domain
from gaussed.utils.shape_helpers import _prod, canonicalise_K_axes, _restore, _as_2d
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
    def __call__(self, ks, F, G, ctx):
        # Raw: (nF, nG, *L, *R)
        K = F.kernel(G, ks, ctx)

        ksL  = _lift_left_all(F.probes[0].ops, ks, ctx)
        ksLR = _lift_right_all(G.probes[0].ops, ksL, ctx)
        L_shape = tuple(getattr(ksLR, "left_shape", ())  or ())
        R_shape = tuple(getattr(ksLR, "right_shape", ()) or ())

        Kc = canonicalise_K_axes(K, L_shape, R_shape)   # (nF, *L, nG, *R)

        nF, nG = int(K.shape[0]), int(K.shape[1])
        Ls, Rs = _prod(L_shape), _prod(R_shape)
        K2 = Kc.reshape(nF * Ls, nG * Rs)               # rows=test, cols=train

        def _as_2d(v): return (v[:, None], True) if v.ndim == 1 else (v, False)
        def _restore(y, s): return y.squeeze(-1) if s else y
        def mv(v): V2, s = _as_2d(v);  return _restore(K2 @ V2, s)
        def rmv(w): W2, s = _as_2d(w); return _restore(K2.T @ W2, s)
        return LinearOp((nF * Ls, nG * Rs), mv=mv, rmv=rmv, to_dense=lambda: K2)

    def tree_flatten(self): return (), ()
    @classmethod
    def tree_unflatten(cls, aux, children): return cls()

# ---- MV-only: build mv/rmv with vmaps/einsums (no full K materialisation) ---
@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class LazyGramConstructor:
    """Non-dense LinearOp: matvecs via kernel contractions (no 2D materialisation)."""
    def __call__(self, ks, F, G, ctx):
        # Read shapes once (Python ints -> JIT-stable)
        ksL  = _lift_left_all(F.probes[0].ops, ks, ctx)
        ksLR = _lift_right_all(G.probes[0].ops, ksL, ctx)
        L_shape = tuple(getattr(ksLR, "left_shape", ())  or ())
        R_shape = tuple(getattr(ksLR, "right_shape", ()) or ())
        Ls, Rs = _prod(L_shape), _prod(R_shape)

        def mv(v):
            V2, was_vec = _as_2d(v)                   # (nG*Rs, k)
            # Infer nF, nG from a single K build (once per call)
            K = F.kernel(G, ks, ctx)                  # (nF, nG, *L, *R)
            nF, nG = int(K.shape[0]), int(K.shape[1])

            V = V2.reshape(nG, *R_shape, -1)          # (nG, *R, k)
            Kc = canonicalise_K_axes(K, L_shape, R_shape)  # (nF, *L, nG, *R)

            # Contract over (nG, *R): result (nF, *L, k)
            lrank, rrank = len(L_shape), len(R_shape)
            axes_K = (1 + lrank,) + tuple(range(2 + lrank, 2 + lrank + rrank))
            axes_V = (0,) + tuple(range(1, 1 + rrank))
            Y = jnp.tensordot(Kc, V, axes=(axes_K, axes_V))   # (nF, *L, k)

            Y2 = Y.reshape(nF * Ls, -1)               # (nF*Ls, k)
            return _restore(Y2, was_vec)

        def rmv(w):
            W2, was_vec = _as_2d(w)                   # (nF*Ls, k)
            K = F.kernel(G, ks, ctx)                  # (nF, nG, *L, *R)
            nF, nG = int(K.shape[0]), int(K.shape[1])

            W = W2.reshape(nF, *L_shape, -1)          # (nF, *L, k)
            Kc = canonicalise_K_axes(K, L_shape, R_shape)  # (nF, *L, nG, *R)

            # Contract over (nF, *L): result (nG, *R, k)
            lrank, rrank = len(L_shape), len(R_shape)
            axes_K = (0,) + tuple(range(1, 1 + lrank))
            axes_W = (0,) + tuple(range(1, 1 + lrank))
            Z = jnp.tensordot(Kc, W, axes=(axes_K, axes_W))  # (nG, *R, k)

            Z2 = Z.reshape(nG * Rs, -1)               # (nG*Rs, k)
            return _restore(Z2, was_vec)

        # TODO: Can implement to_dense lazily (build once per call).
        def to_dense():
            K = F.kernel(G, ks, ctx)                  # (nF, nG, *L, *R)
            nF, nG = int(K.shape[0]), int(K.shape[1])
            Kc = canonicalise_K_axes(K, L_shape, R_shape)
            return Kc.reshape(nF * Ls, nG * Rs)

        # And you need the operator shape:
        # infer nF,nG with a tiny probe (or from F/G meta if you store those)
        K_probe = F.kernel(G, ks, ctx)
        nF, nG = int(K_probe.shape[0]), int(K_probe.shape[1])

        return LinearOp((nF * Ls, nG * Rs), mv=mv, rmv=rmv, to_dense=to_dense)

    def tree_flatten(self): return (), ()
    @classmethod
    def tree_unflatten(cls, aux, children): return cls()
