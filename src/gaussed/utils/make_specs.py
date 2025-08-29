from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Callable, Optional
from jax import Array
import jax


from gaussed.gp.kernels.base import Kernel
from gaussed.gp.means import MeanFun
from gaussed.domains.base import Domain
from gaussed.gp.gp_ops.base import KernelSpec, FunSpec
from gaussed.utils.shape_helpers import _ensure_n_by_d


def make_kernel_spec(
    kernel: Kernel,                     # e.g. an RBFKernel instance with __call__(x,y,domain)
    domain: Domain,
) -> KernelSpec:
    # Fast path: kernel already has a batched __call__(X,Y,domain)
    if hasattr(kernel, "__call__"):
        def k0(X: Array, Y: Array) -> Array:
            return kernel(_ensure_n_by_d(X), _ensure_n_by_d(Y), domain)
    else:
        # Fallback: double vmap over scalar pair(x,y,domain)
        if not hasattr(kernel, "pair"):
            raise TypeError("Kernel must implement __call__ or pair(x,y,domain).")

        def k0(X: Array, Y: Array) -> Array:
            Xn, Yn = _ensure_n_by_d(X), _ensure_n_by_d(Y)
            def row_map(x):
                return jax.vmap(lambda y: kernel.pair(x, y, domain))(Yn)  # (n_y, *L, *R)
            return jax.vmap(row_map)(Xn)  # (n_x, n_y, *L, *R)

    # Pass through factories if present; integrate methods return Optional
    ix_of  = getattr(kernel, "integrate_x_of", None)
    iy_of  = getattr(kernel, "integrate_y_of", None)
    ixy_of = getattr(kernel, "integrate_xy_of", None)

    # Derivatives already bound to base `domain` inside kernel methods (if any)
    d_dx  = getattr(kernel, "d_dx", None)
    d_dy  = getattr(kernel, "d_dy", None)
    d2_xx = getattr(kernel, "d2_xx", None)
    d2_yy = getattr(kernel, "d2_yy", None)
    d2_xy = getattr(kernel, "d2_xy", None)

    return KernelSpec(
        domain=domain,
        k0=k0,
        left_shape=kernel.left_shape,
        right_shape=kernel.right_shape,
        integrate_x_of=ix_of,
        integrate_y_of=iy_of,
        integrate_xy_of=ixy_of,
        d_dx=d_dx, d_dy=d_dy, d2_xx=d2_xx, d2_yy=d2_yy, d2_xy=d2_xy,
    )


def make_fun_spec(
    mean: MeanFun,                   
    domain: Domain,
) -> FunSpec:
    def eval(X: Array) -> Array:
        return mean(X, domain)

    # Pass through factories if present; integrate method returns Optional
    ix_of  = getattr(mean, "integrate", None)

    # Derivatives already bound to base `domain` inside MeanFunc methods (if any)
    partial  = getattr(mean, "partial", None)
    partial2  = getattr(mean, "partial2", None)

    return FunSpec(
        eval=eval,
        integrate=ix_of,
        partial=partial,
        partial2=partial2,
    )