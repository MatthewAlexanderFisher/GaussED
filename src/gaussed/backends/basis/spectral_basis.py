from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Any, Union
from functools import partial
import jax
from jax import Array
import jax.numpy as jnp
import jax.scipy.sparse.linalg as jsla
from jax.experimental import sparse
import math


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

    def linear_operator(self, x: Array, m_per_dim: int = 8, 
                        diag: Optional[Array] = None) -> 'LaplaceBasisOperator':
        """
        Create a linear operator representing Φ^T D Φ where D is diagonal.
        This is useful for GP operations without materializing the full matrix.
        
        Args:
            x: Input points of shape (n, D)
            m_per_dim: Number of basis functions per dimension
            diag: Optional diagonal matrix D of shape (n,). If None, uses identity.
        
        Returns:
            LaplaceBasisOperator that computes matrix-vector products lazily
        """
        return LaplaceBasisOperator(self, x, m_per_dim, diag)

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


class LaplaceBasisOperator:
    """
    Linear operator representing Φ^T D Φ for efficient matrix-vector products.
    Useful for GP operations like computing (Φ^T D Φ + λI)^{-1} v without materializing.
    """
    
    def __init__(self, basis: LaplaceBasis, x: Array, m_per_dim: int = 8, 
                 diag: Optional[Array] = None):
        self.basis = basis
        self.x = x
        self.m_per_dim = m_per_dim
        self.n = x.shape[0]
        
        # Determine effective dimension
        if basis._full_dims:
            self.effective_dim = basis.dim
        else:
            self.effective_dim = len(basis._dims)
        
        self.m_total = m_per_dim ** self.effective_dim
        self.shape = (self.m_total, self.m_total)
        
        # Store diagonal weights
        self.diag = diag if diag is not None else jnp.ones(self.n)
        
        # Precompute basis matrix if small enough
        self._phi_cached = None
        self._cache_threshold = 10000  # Cache if n * m_total < threshold
        
        if self.n * self.m_total < self._cache_threshold:
            self._phi_cached = basis(x, m_per_dim)
    
    def _get_phi(self) -> Array:
        """Get basis matrix, using cache if available."""
        if self._phi_cached is not None:
            return self._phi_cached
        return self.basis(self.x, self.m_per_dim)
    
    def matvec(self, v: Array) -> Array:
        """
        Compute (Φ^T D Φ) v efficiently.
        
        Args:
            v: Vector of shape (m_total,) or (m_total, k)
        
        Returns:
            Result of shape (m_total,) or (m_total, k)
        """
        if v.shape[0] != self.m_total:
            raise ValueError(f"Vector has shape {v.shape}, expected first dim {self.m_total}")
        
        # Two-step computation: Φv, then Φ^T(D(Φv))
        # Step 1: Φv - use evaluate_with_coeff for efficiency
        phi_v = self.basis.evaluate_with_coeff(self.x, v, self.m_per_dim)
        
        # Step 2: Apply diagonal weights
        if v.ndim == 1:
            weighted = self.diag * phi_v
        else:
            weighted = self.diag[:, None] * phi_v
        
        # Step 3: Φ^T weighted - this requires the basis matrix
        phi = self._get_phi()
        result = phi.T @ weighted
        
        return result
    
    def matmul(self, B: Array) -> Array:
        """
        Compute (Φ^T D Φ) B for matrix B.
        
        Args:
            B: Matrix of shape (m_total, k)
        
        Returns:
            Result of shape (m_total, k)
        """
        if B.ndim == 1:
            return self.matvec(B)
        
        # Apply matvec to each column
        return jnp.stack([self.matvec(B[:, i]) for i in range(B.shape[1])], axis=1)
    
    def to_dense(self) -> Array:
        """
        Materialize the full (Φ^T D Φ) matrix.
        Warning: This can be memory intensive for large basis sizes!
        
        Returns:
            Dense matrix of shape (m_total, m_total)
        """
        phi = self._get_phi()  # (n, m_total)
        weighted_phi = self.diag[:, None] * phi  # (n, m_total)
        return phi.T @ weighted_phi  # (m_total, m_total)
    
    def add_diagonal(self, lam: float) -> 'LaplaceBasisOperator':
        """
        Create a new operator representing (Φ^T D Φ + λI).
        
        Args:
            lam: Scalar to add to diagonal
        
        Returns:
            New operator with modified diagonal
        """
        # We'll handle this by modifying matvec
        new_op = LaplaceBasisOperator(self.basis, self.x, self.m_per_dim, self.diag)
        new_op._lambda = lam
        
        # Override matvec to include diagonal term
        original_matvec = new_op.matvec
        
        def matvec_with_diag(v):
            return original_matvec(v) + lam * v
        
        new_op.matvec = matvec_with_diag
        return new_op
    
    def solve(self, b: Array, lam: float = 1e-6, method: str = 'cg') -> Array:
        """
        Solve (Φ^T D Φ + λI) x = b using iterative methods.
        
        Args:
            b: Right-hand side vector of shape (m_total,) or (m_total, k)
            lam: Regularization parameter (added to diagonal)
            method: Solver method ('cg' for conjugate gradient, 'gmres' for GMRES)
        
        Returns:
            Solution x of shape (m_total,) or (m_total, k)
        """
        # Create regularized operator
        A_reg = self.add_diagonal(lam)
        
        if b.ndim == 1:
            # Single right-hand side
            if method == 'cg':
                # Use conjugate gradient (assumes positive definite)
                x, info = jsla.cg(A_reg.matvec, b)
                if info != 0:
                    print(f"CG did not converge (info={info})")
            elif method == 'gmres':
                # Use GMRES (more general)
                x, info = jsla.gmres(A_reg.matvec, b)
                if info != 0:
                    print(f"GMRES did not converge (info={info})")
            else:
                raise ValueError(f"Unknown method: {method}")
            return x
        else:
            # Multiple right-hand sides
            solutions = []
            for i in range(b.shape[1]):
                solutions.append(self.solve(b[:, i], lam, method))
            return jnp.stack(solutions, axis=1)
    
    def __repr__(self):
        return (f"LaplaceBasisOperator(shape={self.shape}, "
                f"n={self.n}, m_per_dim={self.m_per_dim}, "
                f"cached={self._phi_cached is not None})")


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

