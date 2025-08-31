from dataclasses import dataclass, field
from typing import Tuple, Any
import jax
import jax.numpy as jnp
from jax import Array

from gaussed.domains.base import Domain
from gaussed.utils.constraints import Positive, Transform

# ---------- Params ----------
@jax.tree_util.register_pytree_node_class
@dataclass
class CoregionParams:
    # Parameters for the scalar base kernel (e.g., RBFParams/IMQParams/RQParams)
    base_params: Any                  # pytree of the base kernel's params
    # Unconstrained matrix to form a Cholesky factor
    L_params: Array                   # (P, P) unconstrained

    def tree_flatten(self):
        return (self.base_params, self.L_params), None

    @classmethod
    def tree_unflatten(cls, aux, ch):
        base_params, L_params = ch
        return cls(base_params, L_params)

    @classmethod
    def axes(cls, base_params_axis, L_axis):
        obj = object.__new__(cls)
        obj.base_params = base_params_axis
        obj.L_params = L_axis
        return obj


# ---------- Kernel ----------
@jax.tree_util.register_pytree_node_class
@dataclass
class CoregionalKernel:
    base_kernel: Any
    params: CoregionParams
    diag_transform: Any = field(default_factory=Positive)
    enable_checks: bool = True

    # derived/static (not pytree children)
    left_shape: Tuple[int, ...]  = field(init=False)
    right_shape: Tuple[int, ...] = field(init=False)

    def __post_init__(self):
        # Infer P from L_params and set shapes accordingly
        P = int(self.params.L_params.shape[0])
        object.__setattr__(self, "left_shape",  (P,))
        object.__setattr__(self, "right_shape", (P,))
        if self.enable_checks:
            # Ensure base kernel is scalar-valued
            lsh = tuple(getattr(self.base_kernel, "left_shape",  (1,)))
            rsh = tuple(getattr(self.base_kernel, "right_shape", (1,)))
            if lsh != (1,) or rsh != (1,):
                raise ValueError(
                    "[CoregionalKernel] base_kernel must be scalar-valued "
                    f"(left_shape=right_shape=(1,)), got {lsh}, {rsh}"
                )

    # helpers
    def _make_L(self) -> Array:
        L_raw = jnp.tril(self.params.L_params)
        d = jnp.diag(L_raw)
        d_pos = self.diag_transform.forward(d) + 1e-8
        return L_raw - jnp.diag(d) + jnp.diag(d_pos)

    def _coreg_matrix(self) -> Array:
        L = self._make_L()
        return L @ L.T  # (P,P)

    @staticmethod
    def _with_base_params(base_kernel_obj: Any, new_params: Any) -> Any:
        fields = base_kernel_obj.__dict__.copy()
        fields['params'] = new_params
        return type(base_kernel_obj)(**fields)

    # pair: (in,), (in,) -> (P,P)
    def pair(self, x: Array, y: Array, domain: "Domain") -> Array:
        in_shape = tuple(domain.input_shape)
        K = self.__call__(x.reshape((1, *in_shape)),
                          y.reshape((1, *in_shape)),
                          domain)  # (1,1,P,P)
        return K[0, 0, ...]

    # batched: (n_f,*in) × (n_g,*in) -> (n_f, n_g, P, P)
    def __call__(self, x: Array, y: Array, domain: "Domain") -> Array:
        x = domain.ensure_inputs(x)
        y = domain.ensure_inputs(y)
        n_f, n_g = x.shape[0], y.shape[0]

        base_k = self._with_base_params(self.base_kernel, self.params.base_params)
        Kx = base_k(x, y, domain).reshape(n_f, n_g)    # (n_f, n_g)
        B = self._coreg_matrix()                       # (P, P)
        K = Kx[:, :, None, None] * B[None, None, :, :] # (n_f, n_g, P, P)
        return K  # already (n_f, n_g, *left_shape, *right_shape)

    # pytree
    def tree_flatten(self):
        return (self.base_kernel, self.params, self.diag_transform), (self.enable_checks,)

    @classmethod
    def tree_unflatten(cls, aux, children):
        (enable_checks,) = aux
        base_kernel, params, diag_t = children
        obj = cls(base_kernel, params, diag_t, enable_checks)
        return obj

    @classmethod
    def axes(cls, base_kernel_axis: Any, params_axis: CoregionParams, enable_checks: bool = True):
        obj = object.__new__(cls)
        obj.base_kernel = base_kernel_axis
        obj.params = params_axis
        obj.diag_transform = Positive()
        obj.enable_checks = enable_checks
        # derive shapes from params_axis.L_params
        P = int(params_axis.L_params.shape[0])
        obj.left_shape = (P,)
        obj.right_shape = (P,)
        return obj
