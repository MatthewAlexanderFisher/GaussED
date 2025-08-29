from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Callable, Any
import jax
import jax.numpy as jnp
from jax import tree_util as jtu
from jax import Array

from gaussed.domains.base import Domain
from gaussed.utils.geometry import pairwise_r

@jax.tree_util.register_pytree_node_class
@dataclass(init=False)
class Line:
    x1: Array   # (d,)
    x2: Array   # (d,)
    eps: float

    def __init__(self, x1: Array, x2: Array, eps: float = 0.0):
        x1 = jnp.asarray(x1)
        x2 = jnp.asarray(x2)
        if x1.ndim != 1 or x2.ndim != 1 or x1.shape != x2.shape:
            raise ValueError(f"`x1` and `x2` must be 1D with same shape; got {x1.shape=} {x2.shape=}.")
        self.x1 = x1
        self.x2 = x2
        self.eps = float(eps)

    # --- API parity ---
    @property
    def input_shape(self) -> Tuple[int, ...]:
        # ambient space dimension
        return (int(self.x1.shape[0]),)

    def ensure_inputs(self, X: Array) -> Array:
        """Coerce to (n, d) with d = input_shape[0]."""
        X = jnp.asarray(X)
        d = self.x1.shape[0]
        if X.ndim == 1 and d == 1:          # (n,) -> (n,1) when d=1
            return X[:, None]
        if X.ndim == 1 and X.shape[0] == d: # (d,) -> (1,d)
            return X.reshape(1, d)
        if X.ndim == 2 and X.shape[1] == d: # (n,d)
            return X
        raise ValueError(f"Expected (n,{(d,)}) or ({(d,)},) but got {X.shape}")

    def _project_points(self, X: Array) -> Array:
        """Project (n,d) points orthogonally onto the closed segment [x1,x2]."""
        X = self.ensure_inputs(X)           # (n,d)
        v = self.x2 - self.x1               # (d,)
        vv = jnp.dot(v, v)                  # scalar
        # If degenerate segment, collapse to x1
        def _proj_nondeg(_):
            t = jnp.sum((X - self.x1) * v, axis=1) / vv          # (n,)
            t = jnp.clip(t, 0.0, 1.0)
            return self.x1[None, :] + t[:, None] * v[None, :]    # (n,d)
        return jax.lax.cond(vv > 0.0, _proj_nondeg, lambda _: jnp.tile(self.x1, (X.shape[0], 1)), operand=None)

    def project(self, x: Array) -> Array:
        """Public projection API; returns (n,d)."""
        return self._project_points(x)

    def pairwise_geometry(self, X: Array, Y: Array, *, robust: bool = False) -> Array:
        """Euclidean distances between projected points on the segment."""
        Xp = self.project(X)   # (n_x, d)
        Yp = self.project(Y)   # (n_y, d)
        if robust:
            # robust scalar pairwise distances via double vmap over pairwise_r
            def row_map(x):  # x: (d,)
                return jax.vmap(lambda y: pairwise_r(x, y, self.eps))(Yp)
            return jax.vmap(row_map)(Xp)  # (n_x, n_y)
        # fast cdist-style path
        XX = jnp.sum(Xp * Xp, axis=1, keepdims=True)        # (n_x,1)
        YY = jnp.sum(Yp * Yp, axis=1, keepdims=True).T      # (1,n_y)
        sq = jnp.clip(XX + YY - 2.0 * (Xp @ Yp.T), a_min=0.0)
        return jnp.sqrt(sq + self.eps)                      # (n_x, n_y)

    def default_nodes(self, n: int) -> Optional[Array]:
        """Uniform samples along the segment endpoints inclusive; shape (n,d)."""
        t = jnp.linspace(0.0, 1.0, int(n))[:, None]         # (n,1)
        v = (self.x2 - self.x1)[None, :]                    # (1,d)
        return self.x1[None, :] + t * v                     # (n,d)

    # --- PyTree: treat config as static aux (JIT-friendly) ---
    def tree_flatten(self):
        children = ()
        aux = (self.x1, self.x2, self.eps)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        x1, x2, eps = aux
        return cls(x1, x2, eps)

    @classmethod
    def axes(cls, x1_axis, x2_axis, eps_axis):
        obj = object.__new__(cls)
        obj.x1 = x1_axis
        obj.x2 = x2_axis
        obj.eps = eps_axis
        return obj
