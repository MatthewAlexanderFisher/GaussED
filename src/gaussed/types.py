
from __future__ import annotations
from typing import Union, Callable, TypeVar
from jax import Array

from gaussed.gp.gp_ops.probe import Probe, ProbeStack
from gaussed.linops import LinearOp

ProbeLike = Union[Probe, Array, ProbeStack]

# A function over the base variable returning m outputs per input-array
Func = Callable[[Array], Array]  # g(X) -> (..., m) where X is (n,d)


LinearLike = Union[Array, LinearOp]
