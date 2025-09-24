from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Any, Union, Protocol
from functools import partial
import jax
from jax import Array
import jax.numpy as jnp
import jax.scipy.sparse.linalg as jsla
from jax.experimental import sparse
import math

from gaussed.gp.kernels.base import Kernel


# Optional protocol for typing only (no runtime dependency)
class HasSpectral(Protocol):
    def spectral_eig(self, omega: Array) -> Array: ...

SpectralLike = Union[Callable[[Array], Array], HasSpectral]

def _resolve_S_omega(spectral: SpectralLike) -> Callable[[Array], Array]:
    """Return a JAX-callable S(omega) from either a callable or an object with .spectral_eig/.spectral_density."""
    # If it's already a plain callable, use it
    if callable(spectral) and not hasattr(spectral, "spectral_density_1d"):
        return spectral  # type: ignore[arg-type]

    # Otherwise, try to pull a method off the object
    fn = getattr(spectral, "spectral_density_1d", None)

    if fn is None:
        raise TypeError(
            "Spectral provider must be either a callable S(omega) or an "
            "object with .spectral_eig(omega) (or .spectral_density)."
        )
    return fn

@dataclass
class LaplaceBasis:
    """
    Multi-dim Laplace eigenfunction basis with normalisation:
      φ_{i}(x) = ∏_{d=1}^D sqrt(2 b_d / π) * sin( i_d (b_d x_d + c_d) )
    where i_d ∈ {1,...,m_per_dim}. Total features = m_per_dim ** D.
    """
    dim: int
    b: Union[float, Array] = jnp.pi
    c: Union[float, Array] = 0.0
    dims: Optional[Tuple[int, ...]] = None  # subset of input dims, default all
    a: Optional[Callable[[Array, Array], Array]] = None  # optional amplitude function

    def __post_init__(self):
        # coerce b, c to (dim,)
        b = jnp.asarray(self.b)
        c = jnp.asarray(self.c)
        if b.ndim == 0: 
            b = jnp.full((self.dim,), b)
        if c.ndim == 0: 
            c = jnp.full((self.dim,), c)
        assert b.shape == (self.dim,) and c.shape == (self.dim,)
        self._b = b
        self._c = c
        self._norm = jnp.sqrt(2.0 * b / jnp.pi)  # (dim,)

        if self.dims is None:
            self._dims = tuple(range(self.dim))
        else:
            self._dims = tuple(sorted(self.dims))
        self._full_dims = (len(self._dims) == self.dim)
        
        # Default amplitude function
        if self.a is None:
            self.a = lambda x, i: jnp.ones((x.shape[0], 1))

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
        assert D >= len(self._dims), f"x has dim {D}, but basis needs at least {len(self._dims)} dims"

        # Select subset of dims if requested
        x_r = x[:, self._dims] if not self._full_dims else x  # (n, d_sel)
        b = self._b[jnp.array(self._dims)] if not self._full_dims else self._b
        c = self._c[jnp.array(self._dims)] if not self._full_dims else self._c
        norm = self._norm[jnp.array(self._dims)] if not self._full_dims else self._norm
        d_sel = x_r.shape[1]

        # All index tuples j in {1..m}^d_sel
        J = self._multi_index(d_sel, m_per_dim)   # (d_sel, m_total)
        m_total = J.shape[1]

        # Compute per-dimension values then product over dims
        def per_dim_val(d_idx):
            x_d = x_r[:, d_idx:d_idx+1]
            term = self._base_func(b[d_idx] * x_d + c[d_idx], J[d_idx, :])  # (n, m_total)
            return norm[d_idx] * term

        vals = jnp.stack([per_dim_val(d) for d in range(d_sel)], axis=0)  # (d_sel, n, m_total)
        phi = jnp.prod(vals, axis=0)  # (n, m_total)
        
        # Apply amplitude function if needed
        J_T = J.T  # (m_total, d_sel) for PyTorch compatibility
        amplitude = self.a(x, J_T)
        if amplitude.ndim == 1 and amplitude.shape[0] == n:
            amplitude = amplitude[:, None]
        if amplitude.shape == (n, 1):
            amplitude = jnp.broadcast_to(amplitude, (n, m_total))
            
        return amplitude * phi

    def lambd(self, m_per_dim: int, spectral: SpectralLike, *, radial: bool = True) -> Array:
        """
        Build spectral weights D_j for the Laplace basis.
        `spectral`: callable S(omega) OR object with .spectral_eig(omega) (or .spectral_density).
        If `radial=True`, ω_j = sqrt(∑_d (i_d b_d)^2); correct for isotropic kernels (RBF/Matérn).
        """
        S = _resolve_S_omega(spectral)  # S: (ω,) -> (D,)

        # Active dims & their b's
        if self._full_dims:
            b = self._b              # (D,)
            d_sel = self.dim
        else:
            idx = jnp.array(self._dims)
            b = self._b[idx]         # (d_sel,)
            d_sel = b.shape[0]

        # Multi-index J over {1..m_per_dim}^d_sel: shape (d_sel, m_total)
        J = self._multi_index(d_sel, m_per_dim)

        if radial:
            # ω_j = || i ∘ b ||_2
            omega = jnp.sqrt(jnp.sum((J * b[:, None])**2, axis=0))  # (m_total,)
            return S(omega)                                         # (m_total,)
        else:
            # If you need separable/product spectra, you can add a second callable S1d and do:
            # omega_per_dim = J * b[:, None]                          # (d_sel, m_total)
            # return jnp.prod(jax.vmap(S1d)(omega_per_dim), axis=0)
            raise NotImplementedError("Non-radial/separable spectrum not implemented; set radial=True or add S1d.")



    def evaluate_with_coeff(self, F: Any, coeff: Array, m_per_dim: int = 8) -> Array:
        """
        Efficiently evaluate f(x) = Σ_j c_j φ_j(x) without materializing the full basis.
        Uses a vectorized approach that's more efficient than scan for most cases.
        
        Args:
            F: Input data, shape (n, D) or object with .x attribute
            coeff: Coefficients, shape (m_per_dim**effective_dim,) or (..., m_per_dim**effective_dim, k)
            m_per_dim: Number of basis functions per dimension
        
        Returns:
            Array of shape (n,) or (n, k) depending on coeff shape
        """
        x = F.x if hasattr(F, "x") else jnp.asarray(F)  # (n, D)
        n = x.shape[0]
        
        # Select subset of dims if requested
        x_r = x[:, self._dims] if not self._full_dims else x
        b = self._b[jnp.array(self._dims)] if not self._full_dims else self._b
        c = self._c[jnp.array(self._dims)] if not self._full_dims else self._c
        norm = self._norm[jnp.array(self._dims)] if not self._full_dims else self._norm
        d_sel = x_r.shape[1]
        
        # Expected number of basis functions
        expected_size = m_per_dim ** d_sel
        
        # Handle different coefficient shapes
        if coeff.ndim == 1:
            assert coeff.shape[0] == expected_size
            coeff_reshaped = coeff.reshape([m_per_dim] * d_sel)
        elif coeff.ndim == 2:
            assert coeff.shape[0] == expected_size
            k = coeff.shape[1]
            coeff_reshaped = coeff.reshape([m_per_dim] * d_sel + [k])
        else:
            raise ValueError(f"coeff must be 1D or 2D, got shape {coeff.shape}")
        
        # Efficient evaluation using einsum
        # We'll compute the tensor product basis evaluation dimension by dimension
        # and contract with coefficients
        
        # For each dimension, compute basis values
        basis_per_dim = []
        for d in range(d_sel):
            x_d = x_r[:, d:d+1]  # (n, 1)
            j_d = jnp.arange(1, m_per_dim + 1).reshape(1, -1)  # (1, m)
            vals_d = norm[d] * self._base_func(b[d] * x_d + c[d], j_d)  # (n, m)
            basis_per_dim.append(vals_d)
        
        # Now we need to compute the tensor product and contract with coefficients
        # This is where einsum shines
        if coeff.ndim == 1:
            # Build einsum string dynamically
            # Example for 3D: "na,nb,nc,abc->n"
            letters = 'abcdefghijklmnopqrstuvwxyz'[:d_sel]
            einsum_str = ','.join([f'n{l}' for l in letters]) + ',' + letters + '->n'
            result = jnp.einsum(einsum_str, *basis_per_dim, coeff_reshaped)
        else:
            # For matrix coefficients: "na,nb,nc,abck->nk"
            letters = 'abcdefghijklmnopqrstuvwxyz'[:d_sel]
            einsum_str = ','.join([f'n{l}' for l in letters]) + ',' + letters + 'k->nk'
            result = jnp.einsum(einsum_str, *basis_per_dim, coeff_reshaped)
        
        # Apply amplitude function if needed
        J = self._multi_index(d_sel, m_per_dim).T  # (m_total, d_sel)
        amplitude = self.a(x, J)
        if amplitude.shape != (n, expected_size):
            # If amplitude returns scalar or (n, 1), broadcast appropriately
            if amplitude.ndim == 0 or (amplitude.ndim == 2 and amplitude.shape[1] == 1):
                return result  # No amplitude scaling needed beyond constant
        
        # If amplitude varies by basis function, we need to include it in the sum
        # This is more complex and might require materializing more arrays
        # For now, assume amplitude is constant or per-x only
        
        return result


    def set_domain(self, domain: Array) -> 'LaplaceBasis':
        """
        Adjust b and c parameters to match the specified domain.
        
        Args:
            domain: Array of shape (dim, 2) with [lower, upper] bounds
        
        Returns:
            New LaplaceBasis instance with adjusted parameters
        """
        if not self._full_dims:
            relative_domain = domain[jnp.array(self._dims)]
        else:
            relative_domain = domain
        
        new_b = jnp.pi / (relative_domain[:, 1] - relative_domain[:, 0])
        new_c = -relative_domain[:, 0] * new_b
        
        return LaplaceBasis(
            dim=self.dim,
            b=new_b,
            c=new_c,
            dims=self.dims,
            a=self.a
        )



# Helper function for derivative basis functions
def diff_basis_func_gen(order: int):
    """
    Generate basis function for given derivative order.
    Follows the pattern: sin, cos, -sin, -cos, ...
    """
    order = int(order) % 4
    if order == 0:
        return lambda x, i: jnp.sin(i * x)
    elif order == 1:
        return lambda x, i: i * jnp.cos(i * x)  # Include i factor for derivative
    elif order == 2:
        return lambda x, i: -(i**2) * jnp.sin(i * x)  # Include i^2 factor
    elif order == 3:
        return lambda x, i: -(i**3) * jnp.cos(i * x)  # Include i^3 factor

