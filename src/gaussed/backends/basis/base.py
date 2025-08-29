from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Callable, Protocol

from gaussed.types import ProbeLike
# from gaussed.gp.gp_ops.probe import ProbeStack, as_stack


@dataclass
class BasisMap(Protocol):
    """Return design Φ_F for a probe F (n×m). Can be dense or LinearOp."""
    def design(self, F: ProbeLike) -> ProbeLike: ...
    # use as_stack method to coerce Array/Probe into ProbeStack
