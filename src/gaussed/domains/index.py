from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import DTypeLike

@jax.tree_util.register_pytree_node_class
@dataclass(init=False)
class IndexDomain:
    """Finite index set {0,...,n-1}. Pairwise 'distance' is |i-j|."""
    n: int
    geometry_dtype: DTypeLike = field(default=jnp.float32, repr=False)  # static

    def __init__(self, n: int, geometry_dtype: DTypeLike = jnp.float32):
        self.n = int(n)
        self.geometry_dtype = geometry_dtype  # static (not a pytree child)

    @property
    def input_shape(self):
        return (1,)

    def ensure_inputs(self, X: Array) -> Array:
        """Coerce to batch shape (n, *S)."""
        X = jnp.asarray(X)
        S = self.input_shape
        if X.shape == S:                       # (*S,) -> (1,*S)
            return X.reshape((1, *S))
        if S == (1,) and X.ndim == 1:          # (n,) -> (n,1)
            return X[:, None]
        if X.ndim == 1 + len(S) and X.shape[1:] == S:
            return X
        raise ValueError(f"Expected (n,{S}) or {S} but got {X.shape}")


    # --- helpers --------------------------------------------------------------
    def _as_index(self, x: Array) -> Array:
        x = jnp.asarray(x)
        x = jnp.round(x)
        x = jnp.clip(x, 0, self.n - 1)
        return x.astype(jnp.int32).squeeze(-1)  # canonical int indices

    def _cast_geom(self, a: Array) -> Array:
        return a.astype(self.geometry_dtype)

    # --- public API -----------------------------------------------------------
    def project(self, x: Array) -> Array:
        # Keep indices as int32 in [0, n)
        return self._as_index(x)

    def pairwise_geometry(self, x: Array, y: Array) -> Array:
        xi, yj = self._as_index(x), self._as_index(y)
        d = jnp.abs(xi[:, None] - yj[None, :])
        return self._cast_geom(d)

    def squared_pairwise_geometry(self, x: Array, y: Array) -> Array:
        xi, yj = self._as_index(x), self._as_index(y)
        diff = xi[:, None] - yj[None, :]
        return self._cast_geom(diff * diff)

    def default_nodes(self, n: int) -> Optional[Array]:
        return None

    # --- pytree ---------------------------------------------------------------
    def tree_flatten(self):
        # 'n' is dynamic child; dtype is static aux so JIT recompiles only when dtype changes.
        children = (jnp.int32(self.n),)
        aux = (self.geometry_dtype,)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (geometry_dtype,) = aux
        (n_arr,) = children
        return cls(int(n_arr), geometry_dtype=geometry_dtype)

    @classmethod
    def axes(cls, n_axis: int):
        # Optional: axis spec for vmap/pjit
        obj = object.__new__(cls)
        obj.n = n_axis
        obj.geometry_dtype = jnp.float32  # default static
        return obj
