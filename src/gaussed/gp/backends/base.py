from typing import Protocol, Optional, Callable
from dataclasses import dataclass
from jax import Array

from gaussed.gp.base import GP
from gaussed.gp_ops.base import GPOperator, GPProbe
from gaussed.linops.base import LinearOperator

class Backend(Protocol):
    name: str
    def prepare(self, gp: GP, specs: list["Measurement"]) -> "Backend": ...
    def data_operator(self, gp: GP, specs: list["Measurement"]) -> LinearOperator: ...
    def cross_operator_points(self, gp: GP, specs: list["Measurement"], Xt: Array) -> LinearOperator: ...

@dataclass(frozen=True)
class Measurement:
    op: GPOperator
    probe: GPProbe
