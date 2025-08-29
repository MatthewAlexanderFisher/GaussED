from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable, cast
from jax import Array
import jax
import jax.numpy as jnp

# --- small helpers ------------------------------------------------------------

# def _dir_tangent_like(X: Array, axis: int) -> Array:
#     # shape(X) = (n, d). Put ones in the chosen column, zeros elsewhere.
#     return jnp.zeros_like(X).at[:, axis].set(1.0)

def _partial_rows(f: Callable[[Array], Array], axis: int) -> Callable[[Array], Array]:
    # f: (n, d) -> (n, q); returns d/dx_axis f, rowwise, shape (n, q)
    def df(X: Array) -> Array:
        _, d = jax.jvp(f, (X,), (_unit_like(X, axis),))
        return d
    return df

def _partial_rows_x(k: Callable[[Array, Array], Array], axis: int) -> Callable[[Array, Array], Array]:
    # k: (n_x, d), (n_y, d) -> (n_x, n_y); returns d/dx_axis k, shape (n_x, n_y)
    def dk(X: Array, Y: Array) -> Array:
        _, d = jax.jvp(lambda X_: k(X_, Y), (X,), (_unit_like(X, axis),))
        return d
    return dk

def _partial_rows_y(k: Callable[[Array, Array], Array], axis: int) -> Callable[[Array, Array], Array]:
    # returns d/dy_axis k, shape (n_x, n_y)
    def dk(X: Array, Y: Array) -> Array:
        _, d = jax.jvp(lambda Y_: k(X, Y_), (Y,), (_unit_like(Y, axis),))
        return d
    return dk

def _mixed_xy(k: Callable[[Array, Array], Array], ax_x: int, ax_y: int) -> Callable[[Array, Array], Array]:
    # returns d^2/(dx_ax dy_ay) k, shape (n_x, n_y)
    def d2(X: Array, Y: Array) -> Array:
        # first differentiate in x, then in y (order doesn’t matter for common kernels)
        _, dx = jax.jvp(lambda X_: k(X_, Y), (X,), (_unit_like(X, ax_x),))
        _, dxy = jax.jvp(lambda Y_: dx,       (Y,), (_unit_like(Y, ax_y),))
        return dxy
    return d2

# def _dir_tangent_like(X: Array, axis: int) -> Array:
#     # shape(X) = (n, d). Put ones in the chosen column, zeros elsewhere.
#     return jnp.zeros_like(X).at[:, axis].set(1.0)


def _unit_like(X: Array, axis: int) -> Array:
    # shape(X) = (n, d). Put ones in the chosen column, zeros elsewhere.
    return jnp.zeros_like(X).at[:, axis].set(1.0)

# # f: (n,d) -> (n,q)   →  ∂f/∂x_axis, shape (n,q)
# def _partial_rows(f, axis: int):
#     def per_row(x_row: Array) -> Array:                  # (d,) -> (q,)
#         e = _unit_like(x_row, axis)
#         _, d = jax.jvp(lambda xr: f(xr[None, :])[0], (x_row,), (e,))
#         return d
#     return jax.vmap(per_row, in_axes=0)

# # k: (n_x,d),(n_y,d) -> (n_x,n_y)  →  ∂/∂x_axis k, shape (n_x,n_y)
# def _partial_rows_x(k, axis: int):
#     def per_row(x_row: Array, Y: Array) -> Array:        # (d,), (n_y,d) -> (n_y,)
#         e = _unit_like(x_row, axis)
#         _, d = jax.jvp(lambda xr: k(xr[None, :], Y)[0], (x_row,), (e,))
#         return d
#     return lambda X, Y: jax.vmap(per_row, in_axes=(0, None))(X, Y)

# # k: (n_x,d),(n_y,d) -> (n_x,n_y)  →  ∂/∂y_axis k, shape (n_x,n_y)
# def _partial_rows_y(k, axis: int):
#     def per_row(y_row: Array, X: Array) -> Array:        # (d,), (n_x,d) -> (n_x,)
#         e = _unit_like(y_row, axis)
#         _, d = jax.jvp(lambda yr: k(X, yr[None, :])[:, 0], (y_row,), (e,))
#         return d
#     # per_row returns (n_x,) for each y-row ⇒ stack → (n_y,n_x) and **transpose**
#     return lambda X, Y: jax.vmap(per_row, in_axes=(0, None))(Y, X).T

# def _unit_like(v: Array, axis: int) -> Array:
#     e = jnp.zeros_like(v)
#     return e.at[axis].set(1.0)

# # f: (n,d) -> (n,q)  → returns rowwise ∂f/∂x_axis, shape (n,q)
# def _partial_rows(f, axis: int):
#     def per_row(x_row: Array) -> Array:                 # (d,) -> (q,)
#         e = _unit_like(x_row, axis)
#         _, d = jax.jvp(lambda xr: f(xr[None, :])[0], (x_row,), (e,))
#         return d
#     return jax.vmap(per_row, in_axes=0)

# # k: (n_x,d),(n_y,d) -> (n_x,n_y)  → ∂/∂x_axis k, shape (n_x,n_y)
# def _partial_rows_x(k, axis: int):
#     def per_row(x_row: Array, Y: Array) -> Array:       # (d,), (n_y,d) -> (n_y,)
#         e = _unit_like(x_row, axis)
#         _, d = jax.jvp(lambda xr: k(xr[None, :], Y)[0], (x_row,), (e,))
#         return d
#     return lambda X, Y: jax.vmap(per_row, in_axes=(0, None))(X, Y)

# # k: (n_x,d),(n_y,d) -> (n_x,n_y)  → ∂/∂y_axis k, shape (n_x,n_y)
# def _partial_rows_y(k, axis: int):
#     def per_row(y_row: Array, X: Array) -> Array:       # (d,), (n_x,d) -> (n_x,)
#         e = _unit_like(y_row, axis)
#         _, d = jax.jvp(lambda yr: k(X, yr[None, :])[:, 0], (y_row,), (e,))
#         return d
#     # stack columns (n_y, n_x) and transpose to (n_x, n_y)
#     return lambda X, Y: jax.vmap(per_row, in_axes=(0, None))(Y, X).T

# def _mixed_xy(k: Callable[[Array, Array], Array], ax_x: int, ax_y: int):
#     def d2(X: Array, Y: Array) -> Array:
#         def g(Y_):
#             _, dx = jax.jvp(lambda X_: k(X_, Y_), (X,), (_dir_tangent_like(X, ax_x),))
#             return dx
#         _, dxy = jax.jvp(g, (Y,), (_dir_tangent_like(Y, ax_y),))
#         return dxy
#     return d2
