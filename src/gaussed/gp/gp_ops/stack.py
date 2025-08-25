from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Tuple, List
import jax
import jax.numpy as jnp
from jax import Array

from gaussed.gp.gp_ops import Probe  # Protocol with: n(), mean(mean_fn), K_with(other, kernel, domain)

@jax.tree_util.register_pytree_node_class
@dataclass(init=False)
class Stack(Probe):
    """A probe that stacks multiple probes row-wise (block rows).

    If other is also a Stack, K(Stack(parts), Stack(other_parts)) builds
    the full block matrix by calling each child’s K_with.
    """
    parts: Tuple[Probe, ...]  # children

    def __init__(self, parts: Sequence[Probe]):
        assert len(parts) > 0, "Stack must contain at least one probe"
        self.parts = tuple(parts)

    # ---- Probe API ----
    def n(self) -> int:
        # total number of linear functionals
        return int(sum(p.n() for p in self.parts))

    def mean(self, mean_fn) -> Array:
        # concat child means (each shape (n_i,))
        ms = [p.mean(mean_fn) for p in self.parts]
        return jnp.concatenate(ms, axis=0)

    def K_with(self, other: Probe, kernel, domain) -> Array:
        # Case 1: other is a Stack -> build full block matrix
        if isinstance(other, Stack):
            row_blocks = []
            for p in self.parts:
                col_blocks = [p.K_with(q, kernel, domain) for q in other.parts]
                row_blocks.append(jnp.concatenate(col_blocks, axis=1))
            return jnp.concatenate(row_blocks, axis=0)
        # Case 2: other is a single probe -> vertical concatenation of blocks
        blocks = [p.K_with(other, kernel, domain) for p in self.parts]
        return jnp.concatenate(blocks, axis=0)

    # ---- Convenience utilities (handy for noise/y handling) ----
    def block_sizes(self) -> Tuple[int, ...]:
        return tuple(p.n() for p in self.parts)

    def block_slices(self) -> Tuple[slice, ...]:
        sizes = self.block_sizes()
        idx = 0
        out: List[slice] = []
        for m in sizes:
            out.append(slice(idx, idx + m))
            idx += m
        return tuple(out)

    def split(self, v: Array) -> Tuple[Array, ...]:
        """Split a vector v (shape (n(),)) into per-block pieces."""
        return tuple(v[s] for s in self.block_slices())

    # ---- PyTree plumbing ----
    def tree_flatten(self):
        # children are the parts; no static aux needed
        return self.parts, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(children)

    @classmethod
    def axes(cls, *parts_axes: Probe):
        """Build an in_axes for vmap: pass the per-part axes objects."""
        obj = object.__new__(cls)
        obj.parts = tuple(parts_axes)
        return obj
