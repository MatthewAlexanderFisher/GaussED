
from __future__ import annotations
from typing import Union

from gaussed.gp.gp_ops.base import Probe
from jax import Array

ProbeLike = Union[Probe, Array]