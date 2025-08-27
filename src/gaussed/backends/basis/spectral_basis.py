from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Any
from jax import Array
import jax.numpy as jnp

@dataclass
class LaplaceBasis:
    """
    Multi-dim Laplace eigenfunction basis with normalisation:
      φ_{i}(x) = ∏_{d=1}^D sqrt(2 b_d / π) * sin( i_d (b_d x_d + c_d) )
    where i_d ∈ {1,...,m_per_dim}. Total features = m_per_dim ** D.

    This mirrors your torch Laplace.eval (sin branch). If you need the
    cos/± toggling for derivatives, you can plug in a different `func`
    via `diff_basis_func_gen(order)` below.
    """
    dim: int
    b: float = jnp.pi
    c: Array | float = 0.0
    dims: Optional[Tuple[int, ...]] = None  # subset of input dims, default all

    def __post_init__(self):
        # coerce b, c to (dim,)
        b = jnp.asarray(self.b)
        c = jnp.asarray(self.c)
        if b.ndim == 0: b = jnp.full((self.dim,), b)
        if c.ndim == 0: c = jnp.full((self.dim,), c)
        assert b.shape == (self.dim,) and c.shape == (self.dim,)
        self._b = b
        self._c = c
        self._norm = jnp.sqrt(2.0 * b / jnp.pi)  # (dim,)

        if self.dims is None:
            self._dims = tuple(range(self.dim))
        else:
            self._dims = tuple(sorted(self.dims))
        self._full_dims = (len(self._dims) == self.dim)

    @staticmethod
    def _multi_index(dim: int, m_per_dim: int) -> Array:
        """Return shape (dim, m_total) with all index combos in {1..m}^dim."""
        grids = jnp.meshgrid(*[jnp.arange(1, m_per_dim + 1)] * dim, indexing='ij')
        J = jnp.stack(grids, axis=0)      # (dim, m, m, ..., m)
        return J.reshape(dim, m_per_dim ** dim)

    def _base_func(self, x: Array, i: Array) -> Array:
        """sin(i * x) elementwise with broadcasting."""
        return jnp.sin(i * x)

    def __call__(self, F: Any, m_per_dim: int = 8) -> Array:
        """
        Evaluate Φ(F) with Φ_{n,j} = φ_j(x_n).
        F is expected to carry an .x array of shape (n, D) or be the array itself.
        """
        x = F.x if hasattr(F, "x") else jnp.asarray(F)  # (n, D)
        n, D = x.shape
        assert D == self.dim, f"x has dim {D}, but basis.dim = {self.dim}"

        # Select subset of dims if requested
        x_r = x[:, self._dims] if not self._full_dims else x  # (n, d_sel)
        b = self._b[self._dims] if not self._full_dims else self._b
        c = self._c[self._dims] if not self._full_dims else self._c
        norm = self._norm[self._dims] if not self._full_dims else self._norm
        d_sel = x_r.shape[1]

        # All index tuples j in {1..m}^d_sel
        J = self._multi_index(d_sel, m_per_dim)   # (d_sel, m_total)
        m_total = J.shape[1]

        # Compute per-dimension values then product over dims
        # For each dimension, compute: norm_d * sin( j_d * (b_d x_d + c_d) )
        def per_dim_val(d_idx):
            # x_d: (n,1), j_d: (m_total,)
            x_d = x_r[:, d_idx:d_idx+1]
            term = self._base_func(b[d_idx] * x_d + c[d_idx], J[d_idx, :])  # (n, m_total)
            return norm[d_idx] * term

        vals = jnp.stack([per_dim_val(d) for d in range(d_sel)], axis=0)  # (d_sel, n, m_total)
        phi = jnp.prod(vals, axis=0)  # (n, m_total)
        return phi
