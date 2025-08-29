
from __future__ import annotations
from typing import Union, Callable, TypeVar
from jax import Array
from typing import TYPE_CHECKING, Union, Any
from jax import Array

if TYPE_CHECKING:
    from gaussed.linops.linop import LinearOp  # only for type checkers
    from gaussed.gp.gp_ops.probe import Probe, ProbeStack

LinearLike = Union["LinearOp", Array]  # forward-ref; no runtime import
ProbeLike = Union["Probe", Array, "ProbeStack"]

# A function over the base variable returning m outputs per input-array
Func = Callable[[Array], Array]  # g(X) -> (..., m) where X is (n,d)
