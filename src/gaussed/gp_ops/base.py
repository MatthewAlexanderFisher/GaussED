from typing import Protocol, Optional, Tuple, Callable
from jax import Array

from gaussed.gp.kernels import Kernel
from gaussed.gp.means import MeanFunction
from gaussed.linops.base import LinearOperator

class GPOperator(Protocol):
    """Linear operator L acting on functions f: ℝ^d→ℝ."""
    name: str
    # Apply to mean function: x ↦ (L m)(x)
    def apply_mean(self, mean: MeanFunction) -> MeanFunction: ...
    # Apply to kernel: (x,x') ↦ (L_x L_{x'} K)(x,x')
    def apply_kernel(self, kernel: Kernel) -> Kernel: ...
    # Cross-covariances for K_{f, Lf} or K_{Lf, f}
    def apply_kernel_left(self, kernel: Kernel) -> Kernel:  ...   # (L_x K)(x,x')
    def apply_kernel_right(self, kernel: Kernel) -> Kernel: ...   # (L_{x'} K)(x,x')

    def to_basis(self, basis) -> "LinearOperator": ...
    def act_on_feature_map(self, phi: Callable[[Array], Array]) -> Callable[[Array], Array]: ...


class GPProbe(Protocol):
    m: int  # number of outputs

    # default/kernel backend path (works with no basis)
    def apply_mean(self, mean: MeanFunction) -> Array: ...
    def cov_with(self, other: "GPProbe", kernel: Kernel) -> Array: ...
    def cross_with_points(self, X: Array, kernel: Kernel) -> Array: ...

    # adapters for basis/feature backends
    def to_basis(self, basis) -> "LinearOperator": ...        # P: R^n -> R^m
    def to_features(self, phi: Callable[[Array], Array]) -> Array: ...  # A ∈ R^{m×M}