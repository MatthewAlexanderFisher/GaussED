from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple    
import jax
import jax.numpy as jnp
from jax import Array

from gaussed.utils.constraints import Positive, Transform, TriLPositive
from gaussed.engines.linops import LinearOp, ScaledIdentityOp, DiagOp, DenseOp, SumOp, BlockDiagOp

@jax.tree_util.register_pytree_node_class
@dataclass
class NoiseSpec:
    """Gaussian observation noise."""
    def as_op(self, n: int, dtype: Array) -> LinearOp: ...

    def add_to_op(self, A: LinearOp, n: int, dtype) -> LinearOp:
        return SumOp(A, self.as_op(n, dtype))
    
    # Dense (for exact Cholesky) if needed
    def as_matrix(self, n: int, dtype) -> Array:
        return self.as_op(n, dtype).to_dense()


#------- Constant Noise Variance σ * I ------------------------------------------------------
@jax.tree_util.register_pytree_node_class
@dataclass
class HomoskedasticNoise(NoiseSpec):
    raw: Array
    transform: Transform = field(default_factory=Positive)

    def sigma2(self) -> Array: return self.transform.forward(self.raw)

    def as_op(self, n: int, dtype: Array) -> LinearOp:
        return ScaledIdentityOp(n, self.sigma2().astype(dtype), dtype)
    
    def tree_flatten(self): 
        return (self.raw,), (self.transform,)
    
    @classmethod
    def tree_unflatten(cls, aux, ch): 
        (t,) = aux 
        (r,) = ch
        return cls(r, t)

    @classmethod
    def axes(cls, raw_axis):
        obj = object.__new__(cls)
        obj.raw = raw_axis
        obj.transform = Positive()
        return obj


#------- Diagonal Noise Variance Diag(σ1, ..., σn) ------------------------------------------
@jax.tree_util.register_pytree_node_class
@dataclass
class DiagonalNoise(NoiseSpec):
    raw: Array
    transform: Transform = field(default_factory=Positive)

    def diag(self) -> Array: 
        return self.transform.forward(self.raw)

    def as_op(self, n: int, dtype: Array) -> LinearOp:
        d = self.diag().astype(dtype)
        assert d.shape[0] == n
        return DiagOp(d)
    
    def tree_flatten(self): 
        return (self.raw,), (self.transform,)
    
    @classmethod
    def tree_unflatten(cls, aux, ch): 
        (t,) = aux
        (r,) = ch
        return cls(r, t)

    @classmethod
    def axes(cls, raw_axis):
        obj = object.__new__(cls)
        obj.raw = raw_axis
        obj.transform = Positive()
        return obj


#------- Full Noise Variance (dense) Σ ------------------------------------------------------
@jax.tree_util.register_pytree_node_class
@dataclass
class FullNoiseDense(NoiseSpec):
    Sigma: Array
    def as_op(self, n: int, dtype: Array) -> LinearOp:
        S = self.Sigma.astype(dtype)
        assert S.shape == (n, n)
        return DenseOp(S)
    
    def tree_flatten(self):
        return (self.Sigma,), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        Sigma = children
        return cls(Sigma)

    @classmethod
    def axes(cls, S_axis: Array):
        obj = object.__new__(cls)
        obj.Sigma = S_axis
        return obj


#------- Full Noise Variance via Cholesky Σ = L L^T ------------------------------------------------------
@jax.tree_util.register_pytree_node_class
@dataclass
class FullNoiseChol(NoiseSpec):
    raw: Array                    # unconstrained, shape (..., d(d+1)//2)
    tril: TriLPositive            # static (dim, eps), NOT a JAX leaf

    def as_op(self, n: int, dtype: Array) -> LinearOp:
        d = self.tril.dim
        assert n == d, f"FullNoiseChol needs n == dim, got n={n}, dim={d}"
        L = self.tril.forward(self.raw).astype(dtype)  # (..., d, d); typically no batch here
        return LinearOp(
            (d, d),
            mv=lambda v: L @ (L.T @ v),
            rmv=lambda v: L @ (L.T @ v),
            to_dense=lambda: L @ L.T,
        )

    # PyTree: raw is the only leaf; tril is static aux
    def tree_flatten(self):
        return (self.raw,), (self.tril,)

    @classmethod
    def tree_unflatten(cls, aux, children):
        (tril,) = aux
        (raw,) = children
        return cls(raw, tril)

    @classmethod
    def axes(cls, raw_axis, *, tril: TriLPositive):
        """Build an in_axes spec. 'tril' is static (shared across the batch)."""
        obj = object.__new__(cls)
        obj.raw = raw_axis
        obj.tril = tril
        return obj

    # Optional sugar: build from dimension (avoids constructing TriLPositive outside)
    @classmethod
    def axes_with_dim(cls, raw_axis, *, dim: int, eps: float = 1e-12):
        return cls.axes(raw_axis, tril=TriLPositive(dim=dim, eps=eps))

    
#------- Full Noise Variance LinearOp -------------------------------------------------------
@jax.tree_util.register_pytree_node_class
@dataclass
class FullNoiseOp(NoiseSpec):
    op: LinearOp

    def as_op(self, n: int, dtype: Array) -> LinearOp:
        assert self.op.shape == (n, n)
        return self.op
    
    def tree_flatten(self):
        return (self.op, ), None
    
    @classmethod
    def tree_unflatten(cls, aux, children):
        (op,) = children
        return cls(op)

    @classmethod
    def axes(cls, op_axis):
        obj = object.__new__(cls)
        obj.op = op_axis
        return obj


#------- Noiseless (for exact GP interpolation) ---------------------------------------------
@jax.tree_util.register_pytree_node_class
@dataclass
class Noiseless(NoiseSpec):
    """Σ = 0 (exact interpolation)."""
    def as_op(self, n: int, dtype) -> LinearOp:
        return LinearOp((n, n), mv=lambda v: jnp.zeros_like(v),
                        rmv=lambda v: jnp.zeros_like(v),
                        to_dense=lambda: jnp.zeros((n, n), dtype=dtype))
    # PyTree (no leaves)
    def tree_flatten(self): return (), None
    @classmethod
    def tree_unflatten(cls, aux, ch): return cls()

    @classmethod
    def axes(cls):
        return cls()  # nothing to batch


#------- BlockNoise (should be used to cope with different noises based on Probe) -----------
@jax.tree_util.register_pytree_node_class
@dataclass
class BlockNoise(NoiseSpec):
    blocks: Tuple[NoiseSpec, ...]
    sizes: Tuple[int, ...] | None = None  # optional; can infer from F if F is a Stack

    def _sizes_from_probe(self, F) -> Tuple[int, ...]:
        if self.sizes is not None:
            return self.sizes
        # Try to read from your Stack probe helper
        if hasattr(F, "block_sizes"):
            return tuple(F.block_sizes())  # e.g. from your Stack implementation
        raise ValueError("BlockNoise: sizes=None and probe doesn't expose block_sizes().")

    def as_op(self, n: int, dtype) -> LinearOp:
        # If called without a probe context, sizes must be known.
        if self.sizes is None:
            raise ValueError("BlockNoise.as_op requires sizes to be set when no probe is provided.")
        ops = tuple(b.as_op(m, dtype) for b, m in zip(self.blocks, self.sizes))
        op = BlockDiagOp(ops)
        assert op.shape == (n, n), f"BlockNoise op shape {op.shape} != ({n},{n})"
        return op

    # Convenience when you DO have the training probe F (recommended path)
    def as_op_for_probe(self, F, dtype) -> LinearOp:
        sizes = self._sizes_from_probe(F)
        n = sum(sizes)
        ops = tuple(b.as_op(m, dtype) for b, m in zip(self.blocks, sizes))
        return BlockDiagOp(ops)

    def add_to_op(self, A: LinearOp, n: int, dtype) -> LinearOp:
        return SumOp(A, self.as_op(n, dtype))

    def as_matrix(self, n: int, dtype) -> Array:
        return self.as_op(n, dtype).to_dense()

    # PyTree
    def tree_flatten(self):
        # children: blocks (they are pytrees); sizes is static aux
        return (self.blocks,), (self.sizes,)
    
    @classmethod
    def tree_unflatten(cls, aux, children):
        (sizes,) = aux
        (blocks,) = children
        return cls(tuple(blocks), sizes)

    @classmethod
    def axes(cls, blocks_axes: tuple, sizes: Tuple[int, ...] | None = None):
        obj = object.__new__(cls)
        obj.blocks = blocks_axes
        obj.sizes  = sizes
        return obj
