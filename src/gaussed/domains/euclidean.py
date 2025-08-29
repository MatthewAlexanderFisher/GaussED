from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Callable, Any
import jax
import jax.numpy as jnp
from jax import tree_util as jtu
from jax import Array

from gaussed.utils.geometry import pairwise_r
from gaussed.utils.shape_helpers import _ensure_n_by_d

@jax.tree_util.register_pytree_node_class
@dataclass(init=False)
class Euclidean:
    shape: Tuple[int, ...]   # == input_shape
    eps: float
    robust: bool = False

    def __init__(self, input_shape: Tuple[int, ...], eps: float = 0.0, robust: bool = False):
        self.shape = tuple(input_shape)
        self.eps = float(eps)
        self.robust = bool(robust)

    # API
    @property
    def input_shape(self) -> Tuple[int, ...]:
        return self.shape

    def project(self, x: Array) -> Array:
        """Project a single point to the domain; preserve point shape (*S,)."""
        x = jnp.asarray(x)
        S = self.shape
        # Allow scalar → (1,) for 1D input
        if S == (1,) and x.shape == ():
            return x[None]
        if x.shape == S:
            return x
        raise ValueError(f"project expects shape {S} (or scalar for 1D); got {x.shape}")

    def ensure_inputs(self, X: Array) -> Array:
        """Coerce to batch shape (n, *S)."""
        X = jnp.asarray(X)
        S = self.shape
        if X.shape == S:                       # (*S,) -> (1,*S)
            return X.reshape((1, *S))
        if S == (1,) and X.ndim == 1:          # (n,) -> (n,1)
            return X[:, None]
        if X.ndim == 1 + len(S) and X.shape[1:] == S:
            return X
        raise ValueError(f"Expected (n,{S}) or {S} but got {X.shape}")

    def pairwise_geometry(self, x: Array, y: Array) -> Array:
        """Return matrix of pairwise Euclidean distances with optional epsilon."""
        X = self.ensure_inputs(x)
        Y = self.ensure_inputs(y)
        if self.robust:
            # Robust scalar distance for well-behaved JVP at r=0.
            def row_map(xi):  # xi: (*S,)
                return jax.vmap(lambda yj: pairwise_r(xi, yj, self.eps))(Y)
            return jax.vmap(row_map)(X)        # (n_x, n_y)
        # Fast cdist-style computation
        XX = jnp.sum(X * X, axis=1, keepdims=True)        # (n_x,1)
        YY = jnp.sum(Y * Y, axis=1, keepdims=True).T      # (1,n_y)
        sq = jnp.clip(XX + YY - 2.0 * (X @ Y.T), a_min=0.0)
        return jnp.sqrt(sq + self.eps)                    # (n_x, n_y)

    def squared_pairwise_geometry(self, x: Array, y: Array) -> Array:
        """Return matrix of squared Euclidean distances (n_x, n_y)."""
        X = self.ensure_inputs(x)
        Y = self.ensure_inputs(y)

        # Optionally promote low-precision dtypes to float32 for accumulation
        acc_dtype = jnp.float32 if X.dtype in (jnp.float16, jnp.bfloat16) or Y.dtype in (jnp.float16, jnp.bfloat16) else X.dtype
        X = X.astype(acc_dtype)
        Y = Y.astype(acc_dtype)

        if self.robust:
            # Cancellation-free, slightly slower: sum((x_i - y_j)^2, axis=-1)
            diff = X[:, None, :] - Y[None, :, :]          # (n_x, n_y, d)
            sq = jnp.sum(diff * diff, axis=-1)            # (n_x, n_y)
            return sq

        # Fast GEMM path with higher dot precision + clamp
        XX = jnp.sum(X * X, axis=1, keepdims=True)        # (n_x, 1)
        YY = jnp.sum(Y * Y, axis=1, keepdims=True).T      # (1, n_y)

        # Prefer higher precision for the dot to reduce cancellation
        XY = jax.lax.dot_general(
            X, Y,
            dimension_numbers=(((X.ndim - 1,), (Y.ndim - 1,)), ((), ())),
            precision=jax.lax.Precision.HIGHEST
        )  # (n_x, n_y)

        sq = XX + YY - 2.0 * XY
        # Guard small negatives due to roundoff
        return jnp.maximum(sq, 0.0)


    def default_nodes(self, n: int) -> Optional[Array]:
        if self.shape == (1,):
            xs = jnp.linspace(-1.0, 1.0, n)
            return xs[:, None]                            # (n,1)
        return None

    # PyTree: configuration-only
    def tree_flatten(self):
        # Everything is static config; keep in aux
        return (), (self.shape, self.eps, self.robust)

    @classmethod
    def tree_unflatten(cls, aux, children):
        shape, eps, robust = aux
        return cls(shape, eps, robust)

    @classmethod
    def axes(cls, shape_axis, eps_axis, robust_axis=False):
        obj = object.__new__(cls)
        obj.shape = shape_axis
        obj.eps = eps_axis
        obj.robust = robust_axis
        return obj
