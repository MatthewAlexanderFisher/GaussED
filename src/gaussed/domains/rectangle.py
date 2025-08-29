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
class Rectangle:
    bounds: Array  # (d, 2)
    eps: float

    def __init__(self, bounds: Array, eps: float = 0.0):
        b = jnp.asarray(bounds)
        if b.ndim != 2 or b.shape[1] != 2:
            raise ValueError(f"`bounds` must have shape (d,2); got {b.shape}.")
        self.bounds = b
        self.eps = float(eps)

    # --- API parity with Euclidean ---
    @property
    def input_shape(self) -> Tuple[int, ...]:
        return (int(self.bounds.shape[0]),)

    def ensure_inputs(self, X: Array) -> Array:
        """Coerce to (n, d) with d = bounds.shape[0]."""
        X = jnp.asarray(X)
        d = self.bounds.shape[0]
        S = (d,)
        if X.ndim == 1 and d == 1:            # (n,) -> (n,1) when d=1
            return X[:, None]
        if X.ndim == len(S):                   # (d,) -> (1,d)
            if X.shape == S:
                return X.reshape((1, *S))
        if X.ndim == 2 and X.shape[1] == d:    # (n,d)
            return X
        raise ValueError(f"Expected (n,{S}) or ({S},) but got {X.shape}")

    def project(self, x: Array) -> Array:
        """Clip to the hyper-rectangle and ensure (n,d) shape."""
        X = self.ensure_inputs(x)
        lo = self.bounds[:, 0]
        hi = self.bounds[:, 1]
        # broadcast over leading n
        return jnp.clip(X, lo, hi)

    def pairwise_geometry(self, X: Array, Y: Array, *, robust: bool = False) -> Array:
        """Euclidean distances between clipped points inside the rectangle."""
        X = self.project(X)   # (n_x, d)
        Y = self.project(Y)   # (n_y, d)
        if robust:
            # robust scalar pairwise distances via double vmap over pairwise_r
            def row_map(x):  # x: (d,)
                return jax.vmap(lambda y: pairwise_r(x, y, self.eps))(Y)
            return jax.vmap(row_map)(X)  # (n_x, n_y)
        # fast cdist-style path
        XX = jnp.sum(X * X, axis=1, keepdims=True)        # (n_x,1)
        YY = jnp.sum(Y * Y, axis=1, keepdims=True).T      # (1,n_y)
        sq = jnp.clip(XX + YY - 2.0 * (X @ Y.T), a_min=0.0)
        return jnp.sqrt(sq + self.eps)                    # (n_x, n_y)

    def default_nodes(self, n: int) -> Optional[Array]:
        """Axis-aligned tensor grid clipped to bounds; returns (n,d) (first n points)."""
        d = int(self.bounds.shape[0])
        if d == 0:
            return None
        # side length ~ n^(1/d); ceil to cover at least n
        k = jnp.ceil(n ** (1.0 / d)).astype(int)
        # build per-d axes
        axes = [jnp.linspace(self.bounds[i, 0], self.bounds[i, 1], int(k)) for i in range(d)]
        # meshgrid -> (k,...,k,d) -> (k^d, d)
        mesh = jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, d)
        return mesh[:n]

    # --- PyTree: treat config as static aux (matches Euclidean pattern) ---
    def tree_flatten(self):
        children = ()
        aux = (self.bounds, self.eps)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        bounds, eps = aux
        return cls(bounds, eps)

    @classmethod
    def axes(cls, bounds_axis, eps_axis):
        obj = object.__new__(cls)
        obj.bounds = bounds_axis
        obj.eps = eps_axis
        return obj
