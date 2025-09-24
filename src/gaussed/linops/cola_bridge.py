from __future__ import annotations
from typing import Any, Optional, Tuple, Callable
import weakref
import jax.numpy as jnp
from jax import Array

_COLA_REG: "weakref.WeakKeyDictionary[Any, Any]" = weakref.WeakKeyDictionary()

def _try_import_cola():
    try:
        import cola
        return cola
    except Exception:
        return None

def attach_cola(op: Any, cola_op: Any) -> Any:
    """Attach a native CoLA operator to your LinearOp (if cola_op is not None)."""
    if cola_op is not None:
        try:
            _COLA_REG[op] = cola_op
        except TypeError:
            # if the op isn't weak-keyable, just ignore
            pass
    return op

def as_cola(op_or_arr: Any) -> Optional[Any]:
    """Return the native CoLA operator if present or buildable, else None."""
    if isinstance(op_or_arr, Array):
        return None
    # 1) If we attached one in the constructor, return it.
    cola_op = _COLA_REG.get(op_or_arr, None)
    if cola_op is not None:
        return cola_op
    # 2) Last resort: try to wrap from callables (if your LinearOp exposes shape/mv/rmv)
    cola = _try_import_cola()
    if cola is None:
        return None
    if hasattr(op_or_arr, "shape") and hasattr(op_or_arr, "mv") and hasattr(op_or_arr, "rmv"):
        shape: Tuple[int, int] = op_or_arr.shape  # type: ignore[assignment]
        mv: Callable[[Array], Array] = op_or_arr.mv  # type: ignore[assignment]
        rmv: Callable[[Array], Array] = op_or_arr.rmv  # type: ignore[assignment]
        wrapped = cola.ops.LinearOperator(shape=shape, matmul=mv, rmatmul=rmv)
        # If you know it's symmetric, caller can wrap with cola.SelfAdjoint later
        _COLA_REG[op_or_arr] = wrapped
        return wrapped
    return None
