from dataclasses import dataclass
from typing import Tuple
import jax
import jax.numpy as jnp
from jax import Array

from gaussed.gp.gp_ops.base import Operator, Functional, FunSpec, KernelSpec, OpContext, Probe, OpContext

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ProbeStack:
    probes: Tuple[Probe, ...]  # e.g., (F1, F2, ..., Fk)

    # Unary: concatenate rows (first dimension) from each probe's realisation
    def apply_all(self, f: FunSpec, ctx: OpContext) -> Array:
        chunks = tuple(p.apply(f, ctx) for p in self.probes)          # [(n_i, m)]
        # all must share the same m
        return jnp.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]

    # Binary: assemble full block matrix with (i,j) block = probes[i].kernel(probes2[j], ...)
    def kernel_block(self, other: "ProbeStack", ks: KernelSpec, ctx: OpContext) -> Array:
        # First lift ks through each left probe ops (reuse for each right)
        ks_left = tuple(_lift_left_all(p.ops, ks, ctx) for p in self.probes)
        # Then for each left, lift through each right and realise the block via functionals
        blocks: Tuple[Tuple[Array, ...], ...] = tuple(
            tuple(self.probes[i].fnl.pair(other.probes[j].fnl,
                                          _lift_right_all(other.probes[j].ops, ks_left[i], ctx),
                                          ctx)
                  for j in range(len(other.probes)))
            for i in range(len(self.probes))
        )
        # jnp.block expects a nested list/tuple of arrays
        return jnp.block(blocks)

    def tree_flatten(self):
        return ((self.probes,), ())
    @classmethod
    def tree_unflatten(cls, aux, ch):
        (probes,) = ch
        return cls(probes)

# helpers (pure functions so they jit nicely)
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
