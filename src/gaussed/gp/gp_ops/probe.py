from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Optional, Tuple, Callable, Literal, Any
from jax import Array
import jax.numpy as jnp
import jax

from gaussed.gp.gp_ops.base import Operator, Functional, OpContext, FunSpec, KernelSpec
from gaussed.types import ProbeLike

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


# ==== ProbeStack is a stack of Probes ====

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ProbeStack:
    probes: Tuple[Probe, ...]  # (P,)

    # Unary: concatenate rows from each probe's realisation
    def apply(self, f: FunSpec, ctx: OpContext) -> Array:
        if not self.probes:
            raise ValueError("ProbeStack.apply_all: empty probe stack.")
        chunks: Tuple[Array, ...] = tuple(p.apply(f, ctx) for p in self.probes)
        if len(chunks) == 1:
            return chunks[0]
        return jnp.concatenate(chunks, axis=0)

    def kernel(self, other: "ProbeStack", ks: KernelSpec, ctx: OpContext) -> Array:
        if not self.probes or not other.probes:
            raise ValueError("ProbeStack.kernel_block: empty probe stack.")

        ks_left: Tuple[KernelSpec, ...] = tuple(_lift_left_all(p.ops, ks, ctx) for p in self.probes)

        # Build each row (concatenate blocks horizontally)
        row_mats: Tuple[Array, ...] = tuple(
            _concat_row_blocks(tuple(
                self.probes[i].fnl.pair(
                    other.probes[j].fnl,
                    _lift_right_all(other.probes[j].ops, ks_left[i], ctx),
                    ctx,
                )
                for j in range(len(other.probes))
            ))
            for i in range(len(self.probes))
        )

        if len(row_mats) == 1:
            return row_mats[0]
        return jnp.concatenate(row_mats, axis=0)

    # Convenience for self-self blocks
    def gram(self, ks: KernelSpec, ctx: OpContext) -> Array:
        return self.kernel(self, ks, ctx)

    def tree_flatten(self):
        return ((self.probes,), ())

    @classmethod
    def tree_unflatten(cls, aux, ch):
        (probes,) = ch
        return cls(probes)


# ---------- helpers (pure; jit-friendly) ----------

def _lift_left_all(ops: Tuple[Operator, ...], ks: KernelSpec, ctx: OpContext) -> KernelSpec:
    out = ks
    for op in ops:
        out = op.lift_left(out, ctx)
    return out

def _lift_right_all(ops: Tuple[Operator, ...], ks: KernelSpec, ctx: OpContext) -> KernelSpec:
    out = ks
    for op in ops:
        out = op.lift_right(out, ctx)
    return out

def _concat_row_blocks(blocks: Tuple[Array, ...]) -> Array:
    """Concatenate a tuple of 2D blocks horizontally. Assumes consistent row count."""
    if len(blocks) == 0:
        # Should never happen given guards above
        return jnp.zeros((0, 0), dtype=jnp.result_type())
    return blocks[0] if len(blocks) == 1 else jnp.concatenate(blocks, axis=1)

def as_stack(F: ProbeLike) -> ProbeStack:
    return F if isinstance(F, ProbeStack) else (ProbeStack((F,)) if isinstance(F, Probe) else ProbeStack(tuple(F)))

