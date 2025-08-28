from dataclasses import dataclass
from typing import Protocol, Callable
import jax
import jax.numpy as jnp
from jax import Array

from gaussed.linops import LinearOp, DenseOp, AsLinearOp
from gaussed.domains.base import Domain

# ----- Quadrature protocols ---------------------------------------------------
class Quadrature(Protocol):
    # If f: (q,d)->(q,m), should return shape (n_meas, m)
    def apply(self, func: Callable[[Array], Array], dom: Domain) -> Array: ...

class BiQuadrature(Protocol):
    # Numerically compute ∬ F(x,y) dμ_domx(x) dμ_domy(y)
    # If F: (q_x,d_x),(q_y,d_y) -> (q_x,q_y), return (n_L, n_R)
    def apply2(
        self,
        func: Callable[[Array, Array], Array],
        dom_x: Domain,
        dom_y: Domain,
    ) -> Array: ...


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class DiscreteQuadrature:
    nodes: Array    # (q, d)
    weights: Array  # (n_meas, q)

    def apply(self, func: Callable[[Array], Array], dom: Domain) -> Array:
        Fx = func(self.nodes) # (q, m) or (q,)
        Fx = Fx if Fx.ndim == 2 else Fx[:, None]
        return self.weights @ Fx            # (n_meas, m)
    
    def tree_flatten(self): return ((self.nodes, self.weights), ())
    @classmethod
    def tree_unflatten(cls, aux, ch):
        nodes, W = ch
        return cls(nodes, W)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class BiDiscreteQuadrature:
    # Separate X and Y rules
    nodes_x: Array    # (q_x, d_x)
    weights_x: Array  # (n_L, q_x)
    nodes_y: Array    # (q_y, d_y)
    weights_y: Array  # (n_R, q_y)

    def apply2(self, func: Callable[[Array, Array], Array], dom_x: Domain, dom_y: Domain) -> Array:
        Kxy = func(self.nodes_x, self.nodes_y)   # (q_x, q_y)
        return self.weights_x @ Kxy @ self.weights_y.T  # (n_L, n_R)
    
    def tree_flatten(self): return ((self.nodes_x, self.weights_x, self.nodes_y, self.weights_y), ())
    @classmethod
    def tree_unflatten(cls, aux, ch):
        nx, Wx, ny, Wy = ch
        return cls(nx, Wx, ny, Wy)


# TODO: implement get nodes/weights for quadrature routines (domain dependent...)