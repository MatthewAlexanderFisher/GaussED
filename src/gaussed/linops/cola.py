from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Tuple, Union, Optional, Callable
from jax import Array
import jax
import jax.numpy as jnp
from jax import lax

from gaussed.gp.gp_ops.base import KernelSpec, FunSpec, OpContext
from gaussed.utils.shape_helpers import canonicalise_K_axes
from gaussed.gp.gp_ops.probe import _lift_left_all, _lift_right_all, ProbeStack
from gaussed.utils.shape_helpers import _prod
from gaussed.linops.linop import LinearOp


def _try_import_cola():
    try:
        import cola
        return cola
    except Exception:
        return None

def _as_2d(v: Array):
    return (v[:, None], True) if v.ndim == 1 else (v, False)

def _restore(y: Array, squeezed: bool):
    return y.squeeze(-1) if squeezed else y

def _ctx_get_lambda(ctx: Any) -> Array:
    if hasattr(ctx, "basis_lambda"): return getattr(ctx, "basis_lambda")
    if isinstance(ctx, dict) and "basis_lambda" in ctx: return ctx["basis_lambda"]
    return jnp.array(1.0)

def _apply_Lambda(Lam: Array, Z: Array) -> Array:
    if Lam.ndim == 0: return Lam * Z
    if Lam.ndim == 1: return Z * (Lam[:, None] if Z.ndim == 2 else Lam)
    return Lam @ Z

def _basis_dense(DF: Array, DG: Array, Lam: Array) -> Array:
    if Lam.ndim == 0:  return DF @ (Lam * DG.T)
    if Lam.ndim == 1:  return DF @ (DG * Lam[None, :]).T
    return DF @ (Lam @ DG.T)
