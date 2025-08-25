from __future__ import annotations
from typing import Protocol, Optional, Tuple, Callable
from jax import Array

from gaussed.gp.kernels.base import Kernel
from gaussed.gp.means import MeanFunction
from gaussed.engines.linops import LinearOp
from gaussed.domains.base import Domain

class Operator(Protocol):
    """operator L acting on functions f: ℝ^d→ℝ."""
    name: str
    # Apply to mean function: x ↦ (L m)(x)
    def apply_mean(self, mean: MeanFunction) -> MeanFunction: ...
    # Apply to kernel: (x,x') ↦ (L_x L_{x'} K)(x,x')
    def apply_kernel(self, kernel: Kernel) -> Kernel: ...
    # Cross-covariances for K_{f, Lf} or K_{Lf, f}
    def apply_kernel_left(self, kernel: Kernel) -> Kernel:  ...   # (L_x K)(x,x')
    def apply_kernel_right(self, kernel: Kernel) -> Kernel: ...   # (L_{x'} K)(x,x')

    def to_basis(self, basis) -> "LinearOp": ...
    def act_on_feature_map(self, phi: Callable[[Array], Array]) -> Callable[[Array], Array]: ...


class Probe(Protocol):
    """Batch of linear functionals L_i, i=1..n. Knows how to act on k(x,y)."""
    def n(self) -> int: ...
    def mean(self, mean_fn: MeanFunction) -> Array: ...
    def K_with(self, other: "Probe", kernel: "Kernel", domain: Domain) -> Array: ...

    def _K_as_rhs_from_eval(self, X: Array, kernel: Kernel, domain: Domain) -> Array: ...
