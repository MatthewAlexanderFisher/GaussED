from __future__ import annotations
from dataclasses import dataclass
from typing import Union, Protocol, Any, Tuple, Optional, Callable
import jax
import jax.numpy as jnp
from jax import Array, lax
import math

from gaussed.domains.base import Domain
from gaussed.utils.shape_helpers import _prod, canonicalise_K_axes, _restore, _as_2d
from gaussed.linops.linop import LinearOp, DenseOp
from gaussed.linops.cola_bridge import _try_import_cola, attach_cola
from gaussed.gp.gp_ops.probe import ProbeStack, _lift_left_all, _lift_right_all
from gaussed.gp.gp_ops.base import KernelSpec, OpContext, BasisSpec, FunSpec


# ---- Protocol: a constructor that returns a LinearOp ------------------------
class LinOpConstructor(Protocol):
    def __call__(
        self,
        ks_or_phi: Union[KernelSpec, BasisSpec, FunSpec],
        F: ProbeStack,                   # (nF, *input_shape)
        G: ProbeStack,                   # (nG, *input_shape)
        ctx: OpContext
    ) -> LinearOp: ...


# ---- Dense: materialise K, wrap as LinearOp --------------------------------
@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class DenseGramConstructor:
    """
    If ks_or_phi is:
      * KernelSpec: standard kernel path with operator-axis lifting.
      * BasisSpec:  build Φ_F Λ Φ_Gᵀ (square returns efficient LinearOp; cross wrapped as LinearOp).
      * FunSpec:    treated as phi_spec; Λ read from ctx.basis_lambda.
    Always returns a LinearOp whose rows index F and cols index G.
    """

    def __call__(
        self,
        ks_or_phi: Union[KernelSpec, BasisSpec, FunSpec],
        F: ProbeStack,
        G: ProbeStack,
        ctx: OpContext,
    ) -> LinearOp:
        # ===== Basis modes ===================================================
        if isinstance(ks_or_phi, BasisSpec):
            phi_spec, Lam = ks_or_phi.phi_spec, ks_or_phi.Lambda
            return _basis_linearop_always_linearop(phi_spec, Lam, F, G, ctx)

        if isinstance(ks_or_phi, FunSpec):
            Lam = _ctx_get_lambda(ctx)
            return _basis_linearop_always_linearop(ks_or_phi, Lam, F, G, ctx)

        # ===== Kernel mode ===================================================
        ks: KernelSpec = ks_or_phi  # type: ignore[assignment]

        # Raw: (nF, nG, *L, *R)
        K = F.kernel(G, ks, ctx)

        # Operator-lifted shapes
        ksL  = _lift_left_all(F.probes[0].ops, ks, ctx)
        ksLR = _lift_right_all(G.probes[0].ops, ksL, ctx)
        L_shape = tuple(getattr(ksLR, "left_shape", ())  or ())
        R_shape = tuple(getattr(ksLR, "right_shape", ()) or ())

        # Canonical: (nF, *L, nG, *R) -> flatten to (nF*Ls, nG*Rs)
        Kc = canonicalise_K_axes(K, L_shape, R_shape)
        nF, nG = int(K.shape[0]), int(K.shape[1])
        Ls, Rs = _prod(L_shape), _prod(R_shape)
        K2 = Kc.reshape(nF * Ls, nG * Rs)

        def mv(v: Array) -> Array:
            V2, s = _as_2d(v)
            return _restore(K2 @ V2, s)

        def rmv(w: Array) -> Array:
            W2, s = _as_2d(w)
            return _restore(K2.T @ W2, s)

        return LinearOp((nF * Ls, nG * Rs), mv=mv, rmv=rmv, to_dense=lambda: K2)

    def tree_flatten(self):  # stateless
        return (), ()

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls()
    

# ---- CoLa basic constructor (dense K, but CoLa LinearOp) -------------
@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ColaConstructor:
    """
    CoLa constructor that:
      * If passed a FunSpec (basis mode): computes Φ_F Λ Φ_G^T (Λ from ctx.basis_lambda).
      * If passed a KernelSpec: applies K via probe-chunked matvecs; never materialises full K.
    Chunk sizes:
      * ctx.cola_chunk_g : number of G-probes per mv-chunk (default = len(G.probes))
      * ctx.cola_chunk_f : number of F-probes per rmv-chunk (default = len(F.probes))
    """

    def __call__(self,
                 ks_or_phi: Union[KernelSpec, BasisSpec, FunSpec],
                 F: ProbeStack,
                 G: ProbeStack,
                 ctx: OpContext) -> LinearOp:

        # ===== Basis path (FunSpec passed) ==================================
        # 1) BasisSpec(phi_spec, Lambda)
        if isinstance(ks_or_phi, BasisSpec):
            phi_spec, Lam = ks_or_phi.phi_spec, ks_or_phi.Lambda
            return _basis_linearop(phi_spec, Lam, F, G, ctx, use_cola=True)

        # 2) Raw FunSpec (phi_spec) + Lambda from context
        if isinstance(ks_or_phi, FunSpec):
            Lam = _ctx_get_lambda(ctx)
            return _basis_linearop(ks_or_phi, Lam, F, G, ctx, use_cola=True)

        # ===== Kernel path (KernelSpec passed) ==============================
        ks: KernelSpec = ks_or_phi  # type: ignore

        # Operator-axis shapes (static)
        ksL  = _lift_left_all(F.probes[0].ops, ks, ctx)
        ksLR = _lift_right_all(G.probes[0].ops, ksL, ctx)
        L_shape = tuple(getattr(ksLR, "left_shape", ())  or ())
        R_shape = tuple(getattr(ksLR, "right_shape", ()) or ())
        Ls, Rs  = _prod(L_shape), _prod(R_shape)

        P_F = len(F.probes)
        P_G = len(G.probes)

        # Chunk sizes in *probes*
        cg = int(getattr(ctx, "cola_chunk_g", 0) or 0) or P_G

        # We’ll discover these from the first block we build
        def _first_block_dims():
            # build a tiny block to get (nF*Ls, g_chunk*Rs)
            Gc = G.substack(0, min(P_G, 1))
            K_blk = F.kernel(Gc, ks, ctx)                      # (nF, g0, *L, *R)
            K_blk_c = canonicalise_K_axes(K_blk, L_shape, R_shape)  # (nF, *L, g0, *R)
            nF_star = int(K_blk_c.shape[0]) * Ls               # nF * Ls
            g0_star = int(K_blk_c.shape[2]) * Rs               # (rows contributed by Gc) * Rs
            return nF_star, g0_star

        nF_star, g0_star = _first_block_dims()  # row dimension of output, and width of a 1-probe chunk

        # Total width nG*Rs: sum widths over all G-chunks (probe-based)
        def _g_chunk_width(i0, i1) -> int:
            Gc = G.substack(i0, i1)
            # use 1-probe slice of F just to read width; height irrelevant
            Fc = F.substack(0, 1)
            K_blk = Fc.kernel(Gc, ks, ctx)
            K_blk_c = canonicalise_K_axes(K_blk, L_shape, R_shape)
            return int(K_blk_c.shape[2]) * Rs  # cg_rows * Rs

        # --- precompute widths (static) ---------------------------------------------
        widths = []
        Fc1 = F.substack(0, 1)
        for j0 in range(0, P_G, cg):
            j1 = min(P_G, j0 + cg)
            Gc = G.substack(j0, j1)
            K_blk = Fc1.kernel(Gc, ks, ctx)
            K_blk_c = canonicalise_K_axes(K_blk, L_shape, R_shape)
            widths.append(int(K_blk_c.shape[2]) * Rs)
        widths = tuple(int(w) for w in widths)                 # static python tuple[int]
        cuts = tuple(jnp.cumsum(jnp.array(widths, dtype=jnp.int32))[:-1].tolist())
        nG_star = sum(widths)

        # --- mv: y = K @ v ----------------------------------------------------------
        def mv(v: Array) -> Array:
            V2, s = _as_2d(v)                                   # (nG*Rs, k)
            if len(cuts) > 0:
                Vparts = jnp.split(V2, cuts, axis=0)            # list length == len(widths)
            else:
                Vparts = [V2]
            y = jnp.zeros((nF_star, V2.shape[1]), dtype=V2.dtype)

            idx = 0
            for t in range(0, P_G, cg):                         # python-unrolled, static
                t0, t1 = t, min(P_G, t + cg)
                Gc = G.substack(t0, t1)
                K_blk = F.kernel(Gc, ks, ctx)
                K_blk_c = canonicalise_K_axes(K_blk, L_shape, R_shape)
                K2_blk = K_blk_c.reshape(nF_star, -1)           # (nF*Ls, widths[idx])
                y = y + K2_blk @ Vparts[idx]
                idx += 1
            return _restore(y, s)

        # --- rmv: z = K^T @ w -------------------------------------------------------
        def rmv(w: Array) -> Array:
            W2, s = _as_2d(w)                                   # (nF*Ls, k)
            parts = []

            for t in range(0, P_G, cg):                         # python-unrolled, static
                t0, t1 = t, min(P_G, t + cg)
                Gc = G.substack(t0, t1)
                K_blk = F.kernel(Gc, ks, ctx)
                K_blk_c = canonicalise_K_axes(K_blk, L_shape, R_shape)
                K2_blk = K_blk_c.reshape(nF_star, -1)           # (nF*Ls, widths[...])
                parts.append(K2_blk.T @ W2)                      # (widths[...], k)

            z = jnp.concatenate(parts, axis=0) if parts else jnp.zeros((nG_star, W2.shape[1]), dtype=W2.dtype)
            return _restore(z, s)
        
        # Optional dense realisation (chunked over columns)
        def _to_dense():
            I = jnp.eye(nG_star, dtype=jnp.result_type(float))
            cols = []
            step = min(nG_star, 1_000)  # harmless default
            for c0 in range(0, nG_star, step):
                c1 = min(nG_star, c0 + step)
                cols.append(mv(I[:, c0:c1]))
            return jnp.concatenate(cols, axis=1)

        op = LinearOp((nF_star, nG_star), mv=mv, rmv=rmv, to_dense=_to_dense)

        cola = _try_import_cola()
        if cola is not None:
            Kc = cola.ops.LinearOperator(shape=(nF_star, nG_star), matmul=op.mv, rmatmul=op.rmv)
            if (F is G) and (nF_star == nG_star):
                Kc = cola.SelfAdjoint(Kc)
            op = attach_cola(op, Kc)

        return op

    def tree_flatten(self): return (), ()
    @classmethod
    def tree_unflatten(cls, aux, children): return cls()


# ---- MV-only: build mv/rmv with vmaps/einsums (no full K materialisation) ---
@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class StreamingBlockGramConstructor:
    """
    Truly lazy Gram LinearOp using block streaming via Functional.pair.
    No access to 'points' required; works for integrals etc.
    """
    def __call__(self, ks: KernelSpec, F: ProbeStack, G: ProbeStack, ctx: OpContext) -> LinearOp:
        if not F.probes or not G.probes:
            raise ValueError("Empty ProbeStack.")

        # Establish left/right tensor shapes once (operator-lifted)
        ksL0  = _lift_left_all(F.probes[0].ops, ks, ctx)
        ksLR0 = _lift_right_all(G.probes[0].ops, ksL0, ctx)
        L_shape = tuple(getattr(ksLR0, "left_shape", ())  or ())
        R_shape = tuple(getattr(ksLR0, "right_shape", ()) or ())
        lrank, rrank = len(L_shape), len(R_shape)
        Ls, Rs = _prod(L_shape), _prod(R_shape)

        # --- Precompute per-row heights (n_i * Ls) using j=0 as reference
        row_heights: list[int] = []
        n_i_list:   list[int] = []
        for i, pi in enumerate(F.probes):
            ksLi  = _lift_left_all(pi.ops, ks, ctx)
            ksLiR = _lift_right_all(G.probes[0].ops, ksLi, ctx)
            block = pi.fnl.pair(G.probes[0].fnl, ksLiR, ctx)         # (n_i, n_0, *L, *R)
            Bc    = canonicalise_K_axes(block, L_shape, R_shape)     # (n_i, *L, n_0, *R)
            n_i   = int(Bc.shape[0])                                 # rows for this probe
            n_i_list.append(n_i)
            row_heights.append(n_i * Ls)

        # --- Precompute per-col widths (n_j * Rs) using i=0 as reference
        col_widths: list[int] = []
        n_j_list:   list[int] = []
        for j, pj in enumerate(G.probes):
            ksL   = _lift_left_all(F.probes[0].ops, ks, ctx)
            ksLR  = _lift_right_all(pj.ops, ksL, ctx)
            block = F.probes[0].fnl.pair(pj.fnl, ksLR, ctx)          # (n_0, n_j, *L, *R)
            Bc    = canonicalise_K_axes(block, L_shape, R_shape)     # (n_0, *L, n_j, *R)
            n_j   = int(Bc.shape[1 + lrank])                         # column block rows pre-reshape
            n_j_list.append(n_j)
            col_widths.append(n_j * Rs)

        # Prefix offsets for slicing/updates
        row_starts = [0]
        for h in row_heights[:-1]:
            row_starts.append(row_starts[-1] + h)
        col_starts = [0]
        for w in col_widths[:-1]:
            col_starts.append(col_starts[-1] + w)

        nF_tot = sum(row_heights)
        nG_tot = sum(col_widths)

        # ---------- matvec: y = K @ v ----------
        def mv(v: Array) -> Array:
            V2, was_vec = _as_2d(v)                                   # (nG_tot, k)
            kcols = V2.shape[1]
            out = jnp.zeros((nF_tot, kcols), dtype=V2.dtype)

            for i, pi in enumerate(F.probes):
                ksLi = _lift_left_all(pi.ops, ks, ctx)
                n_i  = n_i_list[i]
                acc_i = jnp.zeros((row_heights[i], kcols), dtype=V2.dtype)  # (n_i*Ls, k)

                for j, pj in enumerate(G.probes):
                    ksLiRj = _lift_right_all(pj.ops, ksLi, ctx)
                    # Block K_ij: (n_i, n_j, *L, *R) -> canonical (n_i, *L, n_j, *R)
                    Bij = pi.fnl.pair(pj.fnl, ksLiRj, ctx)
                    Kc  = canonicalise_K_axes(Bij, L_shape, R_shape)

                    # Slice the matching V segment and reshape to (n_j, *R, k)
                    Vseg = lax.dynamic_slice(V2, (col_starts[j], 0), (col_widths[j], kcols))
                    Vblk = Vseg.reshape(n_j_list[j], *R_shape, kcols)

                    # Contract over (n_j, *R): (n_i, *L, k)
                    axes_K = (1 + lrank,) + tuple(range(2 + lrank, 2 + lrank + rrank))
                    axes_V = (0,) + tuple(range(1, 1 + rrank))
                    Y = jnp.tensordot(Kc, Vblk, axes=(axes_K, axes_V))      # (n_i, *L, k)

                    acc_i = acc_i + Y.reshape(row_heights[i], kcols)        # accumulate row-block

                # Write row-block into output
                out = lax.dynamic_update_slice(out, acc_i, (row_starts[i], 0))

            return _restore(out, was_vec)

        # ---------- transpose-matvec: z = K^T @ w ----------
        def rmv(w: Array) -> Array:
            W2, was_vec = _as_2d(w)                                   # (nF_tot, k)
            kcols = W2.shape[1]
            # accumulate per-column block, then scatter into output
            col_accs = [jnp.zeros((col_widths[j], kcols), dtype=W2.dtype) for j in range(len(G.probes))]

            for i, pi in enumerate(F.probes):
                ksLi = _lift_left_all(pi.ops, ks, ctx)
                n_i  = n_i_list[i]
                # Slice row-block of w and reshape to (n_i, *L, k)
                Wseg = lax.dynamic_slice(W2, (row_starts[i], 0), (row_heights[i], kcols))
                Wblk = Wseg.reshape(n_i, *L_shape, kcols)

                for j, pj in enumerate(G.probes):
                    ksLiRj = _lift_right_all(pj.ops, ksLi, ctx)
                    Bij = pi.fnl.pair(pj.fnl, ksLiRj, ctx)
                    Kc  = canonicalise_K_axes(Bij, L_shape, R_shape)        # (n_i, *L, n_j, *R)

                    # Contract over (n_i, *L): -> (n_j, *R, k)
                    axes_K = (0,) + tuple(range(1, 1 + lrank))
                    axes_W = (0,) + tuple(range(1, 1 + lrank))
                    Z = jnp.tensordot(Kc, Wblk, axes=(axes_K, axes_W))      # (n_j, *R, k)

                    col_accs[j] = col_accs[j] + Z.reshape(col_widths[j], kcols)

            out = jnp.zeros((nG_tot, kcols), dtype=W2.dtype)
            for j in range(len(G.probes)):
                out = lax.dynamic_update_slice(out, col_accs[j], (col_starts[j], 0))
            return _restore(out, was_vec)

        # Optional dense materialisation (one pass)
        def to_dense():
            # Build by applying mv to basis vectors in blocks to avoid O(n^2) loops if you like.
            I = jnp.eye(nG_tot, dtype=jnp.float32)
            return jax.vmap(mv, in_axes=1, out_axes=1)(I)

        return LinearOp((nF_tot, nG_tot), mv=mv, rmv=rmv, to_dense=to_dense)

    def tree_flatten(self): return (), ()
    @classmethod
    def tree_unflatten(cls, aux, children): return cls()


# helpers for constructors ---------------------------------------------------

def _ctx_get_lambda(ctx: Any) -> Array:
    """Allow supplying Lambda via context if only a FunSpec was passed."""
    # Try attribute first, then mapping
    if hasattr(ctx, "basis_lambda"):
        return getattr(ctx, "basis_lambda")
    if isinstance(ctx, dict) and "basis_lambda" in ctx:
        return ctx["basis_lambda"]
    # Fallback (safe default; you can assert instead if you prefer)
    return jnp.array(1.0)


def _apply_Lambda(Lam: Array, Z: Array) -> Array:
    # Z: (m,) or (m,k)
    if Lam.ndim == 0:
        return Lam * Z
    elif Lam.ndim == 1:
        # diag vector
        return Z * (Lam[:, None] if Z.ndim == 2 else Lam)
    else:
        # full (m,m)
        return Lam @ Z

def _basis_linearop_always_linearop(phi_spec: FunSpec, Lam: Array, F: ProbeStack, G: ProbeStack, ctx: OpContext):
    DF = F.apply(phi_spec, ctx)  # (nF, m)
    DG = G.apply(phi_spec, ctx)  # (nG, m)

    def _apply_Lambda(L: Array, Z: Array) -> Array:
        if L.ndim == 0:  return L * Z
        if L.ndim == 1:  return Z * (L[:, None] if Z.ndim == 2 else L)
        return L @ Z

    if F is G:
        n, m = DF.shape
        def mv(v: Array) -> Array:
            V2, s = _as_2d(v)
            Z = DF.T @ V2
            Z = _apply_Lambda(Lam, Z)
            Y = DF @ Z
            return _restore(Y, s)
        def rmv(w: Array) -> Array:
            W2, s = _as_2d(w)
            Z = DF.T @ W2
            Z = _apply_Lambda(Lam, Z)
            Y = DF @ Z
            return _restore(Y, s)
        def _to_dense():
            if Lam.ndim == 0:  return DF @ (Lam * DF.T)
            if Lam.ndim == 1:  return DF @ (DF * Lam[None, :]).T
            return DF @ (Lam @ DF.T)
        return LinearOp((n, n), mv=mv, rmv=rmv, to_dense=_to_dense)

    # Cross (wrap dense as LinearOp to keep the type consistent)
    if Lam.ndim == 0:
        K = DF @ (Lam * DG.T)
    elif Lam.ndim == 1:
        K = DF @ (DG * Lam[None, :]).T
    else:
        K = DF @ (Lam @ DG.T)

    nF, nG = int(K.shape[0]), int(K.shape[1])

    def mv(v: Array) -> Array:
        V2, s = _as_2d(v)
        return _restore(K @ V2, s)

    def rmv(w: Array) -> Array:
        W2, s = _as_2d(w)
        return _restore(K.T @ W2, s)

    return LinearOp((nF, nG), mv=mv, rmv=rmv, to_dense=lambda: K)

def _basis_linearop(
    phi_spec: FunSpec,
    Lam: Array,
    F: ProbeStack,
    G: ProbeStack,
    ctx: OpContext,
    use_cola: bool,
    noise_var: float = 0.0,
) -> LinearOp:
    DF = F.apply(phi_spec, ctx)  # (nF, m)
    DG = G.apply(phi_spec, ctx)  # (nG, m)

    def _apply_Lambda(L: Array, Z: Array) -> Array:
        if L.ndim == 0:
            return L * Z
        if L.ndim == 1:
            return Z * (L[:, None] if Z.ndim == 2 else L)
        return L @ Z

    # ---- square block: return efficient LinearOp; optional σ²I and CoLA attach
    if F is G:
        n, m = DF.shape
        sigma2 = jnp.asarray(noise_var, dtype=DF.dtype)

        def mv(v: Array) -> Array:
            V2, s = _as_2d(v)           # (n,k)
            Z = DF.T @ V2               # (m,k)
            Z = _apply_Lambda(Lam, Z)   # (m,k)
            Y = DF @ Z                  # (n,k)
            Y = Y + sigma2 * V2         # add noise (no branch; sigma2 may be 0.)
            return _restore(Y, s)

        def rmv(w: Array) -> Array:
            W2, s = _as_2d(w)           # (n,k)
            Z = DF.T @ W2               # (m,k)
            Z = _apply_Lambda(Lam, Z)   # (m,k)
            Y = DF @ Z                  # (n,k)
            Y = Y + sigma2 * W2
            return _restore(Y, s)

        def _to_dense():
            base = (DF @ (Lam * DF.T)) if Lam.ndim == 0 else (
                   DF @ (DF * Lam[None, :]).T if Lam.ndim == 1 else
                   DF @ (Lam @ DF.T))
            return base + sigma2 * jnp.eye(n, dtype=DF.dtype)

        op = LinearOp((n, n), mv=mv, rmv=rmv, to_dense=_to_dense)

        if use_cola:
            cola = _try_import_cola()
            if cola is not None:
                Phi = cola.ops.Dense(DF)
                if Lam.ndim == 0:
                    Lam_op = cola.ops.Diagonal(jnp.full((m,), Lam, dtype=DF.dtype))
                elif Lam.ndim == 1:
                    Lam_op = cola.ops.Diagonal(Lam)
                else:
                    Lam_op = cola.ops.Dense(Lam)
                Kc = cola.SelfAdjoint(Phi @ Lam_op @ Phi.T)
                if float(noise_var) != 0.0:
                    Kc = Kc + float(noise_var) * cola.ops.I_like(Kc)
                op = attach_cola(op, Kc)

        return op

    # ---- cross block: build dense once, wrap as LinearOp; optional CoLA attach
    if Lam.ndim == 0:
        K = DF @ (Lam * DG.T)                  # (nF, nG)
    elif Lam.ndim == 1:
        K = DF @ (DG * Lam[None, :]).T         # (nF, nG)
    else:
        K = DF @ (Lam @ DG.T)                  # (nF, nG)

    nF, nG = int(K.shape[0]), int(K.shape[1])

    def mv(v: Array) -> Array:
        V2, s = _as_2d(v)                      # (nG,k)
        return _restore(K @ V2, s)

    def rmv(w: Array) -> Array:
        W2, s = _as_2d(w)                      # (nF,k)
        return _restore(K.T @ W2, s)

    op = LinearOp((nF, nG), mv=mv, rmv=rmv, to_dense=lambda: K)

    if use_cola:
        cola = _try_import_cola()
        if cola is not None:
            PhiF = cola.ops.Dense(DF)
            PhiG = cola.ops.Dense(DG)
            if Lam.ndim == 0:
                Lam_op = cola.ops.Diagonal(jnp.full((DF.shape[1],), Lam, dtype=DF.dtype))
            elif Lam.ndim == 1:
                Lam_op = cola.ops.Diagonal(Lam)
            else:
                Lam_op = cola.ops.Dense(Lam)
            Kc = PhiF @ Lam_op @ PhiG.T  # not self-adjoint
            op = attach_cola(op, Kc)

    return op



# ---- CoLa helpers ------------------------------------------------------------

def _try_import_cola():
    try:
        import cola
        return cola
    except Exception:
        return None


def _cola_from_linearop(shape: Tuple[int, int],
                        mv: Callable[[Array], Array],
                        rmv: Callable[[Array], Array],
                        symmetric: bool) -> Optional[Any]:
    cola = _try_import_cola()
    if cola is None:
        return None
    # CoLA's generic LinearOperator from callables:
    Op = cola.ops.LinearOperator(shape=shape,
                                 matmul=mv,
                                 rmatmul=rmv)
    return cola.SelfAdjoint(Op) if symmetric and shape[0] == shape[1] else Op


def _cola_from_basis(DF: Array, Lam: Array, DG: Optional[Array] = None,
                     noise_var: float = 0.0) -> Optional[Any]:
    cola = _try_import_cola()
    if cola is None:
        return None
    PhiF = cola.ops.Dense(DF)
    if DG is None:
        PhiG_T = PhiF.T
    else:
        PhiG_T = cola.ops.Dense(DG).T

    if Lam.ndim == 0:
        Lam_op = cola.ops.Diagonal(jnp.full((DF.shape[1],), Lam))
    elif Lam.ndim == 1:
        Lam_op = cola.ops.Diagonal(Lam)
    else:
        Lam_op = cola.ops.Dense(Lam)

    K = PhiF @ Lam_op @ PhiG_T
    if DG is None:
        K = cola.SelfAdjoint(K)
        if noise_var:
            K = K + noise_var * cola.ops.I_like(K)
    return K
