
from __future__ import annotations
from typing import Union, Callable
from jax import Array

from gaussed.gp.gp_ops.base import Probe
from gaussed.engines.linops import LinearOp

ProbeLike = Union[Probe, Array]

# A function over the base variable returning m outputs per input-array
Func = Callable[[Array], Array]  # g(X) -> (..., m) where X is (n,d)


LinearLike = Union[Array, LinearOp]
