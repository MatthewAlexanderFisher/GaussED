from __future__ import annotations
from typing import Union, TYPE_CHECKING, Protocol, runtime_checkable, Callable
from jax import Array
import jax.numpy as jnp

if TYPE_CHECKING:
    from gaussed.linops.linop import LinearOp  # wrapper (shape, mv, rmv [, to_dense])
    from gaussed.gp.gp_ops.probe import Probe, ProbeStack
    from gaussed.gp.gp_ops.base import FunSpec, BasisSpec, KernelSpec

# Public alias: keep it simple and JAX-only.
LinearLike = Union["LinearOp", Array]
ProbeLike = Union["Probe", Array, "ProbeStack"]

@runtime_checkable
class _MaybeDense(Protocol):
    # Optional capability we can EAFP on
    def to_dense(self) -> Array: ...

def as_dense(K: LinearLike, col_block: int = 2048) -> Array:
    """Materialise a LinearLike to a dense Array in a JIT-friendly way."""
    if isinstance(K, jnp.ndarray):
        return K
    # Try fast path if provided
    if isinstance(K, _MaybeDense) and hasattr(K, "to_dense"):
        return K.to_dense()  # type: ignore[attr-defined]

    # Generic fallback via matvecs: K @ I, in column blocks
    # Expect your LinearOp to expose `shape` and `mv`.
    n_rows, n_cols = K.shape  # type: ignore[union-attr]
    out_cols = []
    I = jnp.eye(n_cols, dtype=jnp.result_type(float))
    for c0 in range(0, n_cols, col_block):
        c1 = min(n_cols, c0 + col_block)
        out_cols.append(K.mv(I[:, c0:c1]))  # type: ignore[attr-defined]
    return jnp.concatenate(out_cols, axis=1)


# A function over the base variable returning m outputs per input-array
Func = Callable[[Array], Array]  # g(X) -> (..., m) where X is (n,d)


FeatureLike = Union["KernelSpec", "FunSpec", "BasisSpec"]