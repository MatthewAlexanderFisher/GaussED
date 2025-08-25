from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Callable, Any
import jax
import jax.numpy as jnp
from jax import Array

from gaussed.domains.base import Domain

@jax.tree_util.register_pytree_node_class
@dataclass(init=False)
class Graph:
    adjacency: Array                 # (N,N)
    distances: Optional[Array]
    nodes_coords: Optional[Array]

    def __init__(self, adjacency: Array, distances: Optional[Array] = None, nodes_coords: Optional[Array] = None):
        self.adjacency = jnp.asarray(adjacency)
        self.distances = None if distances is None else jnp.asarray(distances)
        self.nodes_coords = None if nodes_coords is None else jnp.asarray(nodes_coords)

    @property
    def event_shape(self): return ()  # integer node id
    def project(self, i: Array) -> Array:
        N = self.adjacency.shape[0]
        return jnp.clip(jnp.asarray(i, dtype=jnp.int32), 0, N - 1)

    def pairwise_geometry(self, i: Array, j: Array) -> Array:
        if self.distances is None:
            raise ValueError("Provide Graph.distances for pairwise geometry.")
        i, j = self.project(i), self.project(j)
        return self.distances[i, j]

    def default_nodes(self, n: int) -> Optional[Array]:
        N = self.adjacency.shape[0]
        return jnp.arange(jnp.minimum(n, N))

    def tree_flatten(self):
        return (self.adjacency, self.distances, self.nodes_coords), None
    @classmethod
    def tree_unflatten(cls, aux, children):
        A, D, X = children; return cls(A, D, X)

    @classmethod
    def axes(cls, A_axis, D_axis, X_axis):
        obj = object.__new__(cls); obj.adjacency = A_axis; obj.distances = D_axis; obj.nodes_coords = X_axis; return obj
