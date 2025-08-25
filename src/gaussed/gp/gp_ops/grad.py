from __future__ import annotations
from typing import Protocol, Optional, Tuple, Callable
from jax import Array
import jax.numpy as jnp
import jax

from gaussed.gp.kernels.base import Kernel
from gaussed.domains.base import Domain
from gaussed.gp.gp_ops.base import Operator, Probe


class Grad(Probe):
    def __init__(self, X: Array, axis: int):
        self.X = jnp.asarray(X); self.axis = int(axis)

    def n(self): return self.X.shape[0]
    def mean(self, mean_fn): return mean_fn.grad(self.X, axis=self.axis)  # optional; else AD mean

    def K_with(self, other: Probe, kernel: Kernel, domain: Domain) -> Array:
        return other._K_as_rhs_from_grad(self.X, self.axis, kernel, domain)

    def _K_as_lhs_to_eval(self, Y, kernel, domain) -> Array:
        # K(∂/∂x_a @ X, Eval(Y)) = ∂/∂x_a k(x,y)|_{x=X}
        def kx(x): return jax.vmap(lambda y: kernel(x, y, domain))(Y)      # (|Y|,)
        grad_x = jax.jacfwd(kx)  # gradient wrt x, returns (d, |Y|)
        G = jax.vmap(grad_x)(self.X)                                      # (|X|, d, |Y|)
        return G[:, self.axis, :]                                         # (|X|, |Y|)

    def _K_as_rhs_from_eval(self, X, kernel, domain) -> Array:
        # K(Eval(X), ∂/∂y_a @ Y) = ∂/∂y_a k(x,y)|_{y=Y}
        def ky(y): return jax.vmap(lambda x: kernel(x, y, domain))(X)      # (|X|,)
        grad_y = jax.jacfwd(ky)                                           # (d, |X|)
        G = jax.vmap(grad_y)(self.X)                                      # (|Y|, d, |X|)
        return G[:, self.axis, :].swapaxes(0,1)                            # (|X|, |Y|)

    def _K_as_rhs_from_grad(self, X: Array, axis_l: Array, kernel: Kernel, domain: Domain) -> Array:
        # K(∂/∂x_l @ X, ∂/∂y_a @ Y) = ∂²/∂x_l∂y_a k(x,y)
        def kxy(x, y): return kernel(x, y, domain)
        # Mixed partial via forward-on-x then forward-on-y (or reverse), consistent since k is C^2
        def kx(x):
            return jax.jacfwd(lambda xx: jax.vmap(lambda yy: kxy(xx, yy))(self.X), argnums=0)(x)  # (d, |Y|)
        # gradient wrt y applied after: grad_y( kx(x)[l, :] )[a]
        def mixed(x):
            # grad over Y (axis 0 of the mapped output)
            def g_y(y):
                return jax.jacfwd(lambda yy: kxy(x, yy))(y)  # (d,)
            Gy = jax.vmap(g_y)(self.X)  # (|Y|, d)
            return Gy  # we’ll index later
        # Simpler (and faster) in practice: build explicit second derivative helpers once in kernels.
        raise NotImplementedError  # keep interface; implement properly in your repo
