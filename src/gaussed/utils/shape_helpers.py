"""
Shape manipulation utilities for tensor operations in Gaussian process computations.

This module provides utilities for reshaping and manipulating tensors, particularly
for kernel matrices and vectors with event dimensions.
"""

from typing import Tuple
import math
import jax.numpy as jnp
from jax import Array


# ============================================================================
# Basic Shape Operations
# ============================================================================

def prod(shape: Tuple[int, ...]) -> int:
    """Compute product of shape tuple elements. Returns 1 for empty tuple."""
    return math.prod(shape) if shape else 1


def ensure_2d_column(x: Array) -> Array:
    """Ensure array is 2D column vector: (n,) -> (n, 1)."""
    x = jnp.asarray(x)
    return x.reshape(-1, 1) if x.ndim == 1 else x


def ensure_2d_rows(Y: Array) -> Array:
    """Reshape (..., Q) -> (N, Q) where N is product of leading dims."""
    return Y.reshape((-1, Y.shape[-1]))


def ensure_n_by_shape(X: Array, item_shape: Tuple[int, ...]) -> Array:
    """
    Coerce X to shape (n, *item_shape).
    
    Args:
        X: Input array
        item_shape: Desired shape for each item
        
    Returns:
        Array with shape (n, *item_shape)
        
    Accepts:
        - (n, *item_shape): returns as-is
        - (*item_shape,): treated as n=1, returns (1, *item_shape)
        - (n,) when item_shape=(1,): returns (n, 1)
    """
    X = jnp.asarray(X)
    
    # Single item case: (*item_shape,) -> (1, *item_shape)
    if X.ndim == len(item_shape):
        return X.reshape((1, *item_shape))
    
    # Common scalar case: (n,) -> (n, 1)
    if X.ndim == 1 and item_shape == (1,):
        return X.reshape((-1, 1))
    
    # Already correct shape
    if X.ndim == 1 + len(item_shape) and X.shape[1:] == item_shape:
        return X
    
    raise ValueError(f"Expected (n, {item_shape}), got {X.shape}")


# ============================================================================
# Axis Manipulation
# ============================================================================

def move_front_axis(a: Array, dst: int) -> Array:
    """
    Move leading axis to position `dst`.
    
    Args:
        a: Input array
        dst: Destination position (0 <= dst < a.ndim)
        
    Returns:
        Array with axis 0 moved to position dst
    """
    nd = a.ndim
    assert 0 <= dst < nd, f"Invalid destination {dst} for {nd}D array"
    perm = [*range(1, dst + 1), 0, *range(dst + 1, nd)]
    return jnp.transpose(a, perm)


def canonicalise_kernel_axes(K: Array, 
                            L_shape: Tuple[int, ...], 
                            R_shape: Tuple[int, ...]) -> Array:
    """
    Reorder kernel axes from (nF, nG, *L, *R) to (nF, *L, nG, *R).
    
    This is useful for tensor kernels where we want to group row indices
    together and column indices together.
    
    Args:
        K: Kernel array with shape (nF, nG, *L, *R)
        L_shape: Left event shape
        R_shape: Right event shape
        
    Returns:
        Reordered array with shape (nF, *L, nG, *R)
    """
    nF, nG = int(K.shape[0]), int(K.shape[1])
    lrank, rrank = len(L_shape), len(R_shape)
    
    # Validate input
    expected_ndim = 2 + lrank + rrank
    assert K.ndim == expected_ndim, \
        f"Expected {expected_ndim}D array, got {K.ndim}D with shape {K.shape}"
    
    # No reordering needed for scalar kernels
    if lrank == 0 and rrank == 0:
        return K
    
    # Compute new axis order
    row_axes = (0,) + tuple(range(2, 2 + lrank))                  # (nF, *L)
    col_axes = (1,) + tuple(range(2 + lrank, 2 + lrank + rrank))  # (nG, *R)
    
    return jnp.transpose(K, row_axes + col_axes)


# ============================================================================
# Kernel Matrix Operations
# ============================================================================

def enforce_kernel_shape(K: Array,
                        n_x: int,
                        n_y: int,
                        left_shape: Tuple[int, ...],
                        right_shape: Tuple[int, ...]) -> Array:
    """
    Ensure kernel has shape (n_x, n_y, *left_shape, *right_shape).
    
    Handles common edge cases like scalar outputs and single evaluations.
    
    Args:
        K: Kernel evaluation result
        n_x: Number of x points
        n_y: Number of y points
        left_shape: Left event shape
        right_shape: Right event shape
        
    Returns:
        Kernel with correct shape
    """
    K = jnp.asarray(K)
    want_nd = 2 + len(left_shape) + len(right_shape)
    
    # Already correct
    if K.ndim == want_nd:
        return K
    
    # Case 1: Scalar kernel with correct batch dims
    if K.ndim == 2 and not left_shape and not right_shape:
        return K
    
    # Case 2: Single evaluation, scalar output: () → (1, 1)
    if K.ndim == 0 and n_x == 1 and n_y == 1 and not left_shape and not right_shape:
        return K.reshape(1, 1)
    
    # Case 3: Single evaluation, tensor output: (*L, *R) → (1, 1, *L, *R)
    if K.shape == (*left_shape, *right_shape) and n_x == 1 and n_y == 1:
        return K.reshape((1, 1, *left_shape, *right_shape))
    
    # Case 4: Try reshaping if total size matches
    expected_size = n_x * n_y * prod(left_shape) * prod(right_shape)
    if K.size == expected_size:
        return K.reshape((n_x, n_y, *left_shape, *right_shape))
    
    # Shape mismatch
    raise ValueError(
        f"Cannot reshape kernel from {K.shape} to "
        f"({n_x}, {n_y}, {left_shape}, {right_shape})"
    )


def flatten_kernel_for_solver(K: Array) -> Array:
    """
    Flatten kernel for linear solvers: (n_x, n_y, *L, *R) -> (n_x*|L|, n_y*|R|).
    
    Args:
        K: Kernel array with event dimensions
        
    Returns:
        Flattened kernel matrix suitable for linear algebra operations
    """
    if K.ndim == 2:  # Already flat (scalar kernel)
        return K
    
    n_x, n_y = K.shape[:2]
    mid = (K.ndim - 2) // 2
    left_dims = K.shape[2:2 + mid]
    right_dims = K.shape[2 + mid:]
    
    # Reshape to separate event dimensions
    left_size = prod(left_dims) if left_dims else 1
    right_size = prod(right_dims) if right_dims else 1
    K_reshaped = K.reshape(n_x, n_y, left_size, right_size)
    
    # Transpose and flatten
    K_transposed = jnp.transpose(K_reshaped, (0, 2, 1, 3))
    return K_transposed.reshape(n_x * left_size, n_y * right_size)


# ============================================================================
# Packing and Unpacking Operations
# ============================================================================

def pack_kernel_to_matrix(K: Array,
                         left_shape: Tuple[int, ...],
                         right_shape: Tuple[int, ...]) -> Array:
    """
    Pack kernel with event dims to matrix: (n_x, n_y, *L, *R) -> (n_x*|L|, n_y*|R|).
    
    Args:
        K: Kernel array
        left_shape: Left event shape
        right_shape: Right event shape
        
    Returns:
        Packed matrix
    """
    n_x, n_y = K.shape[:2]
    return K.reshape(n_x * prod(left_shape), n_y * prod(right_shape))


def unpack_matrix_to_kernel(M: Array,
                           n_x: int, 
                           n_y: int,
                           left_shape: Tuple[int, ...],
                           right_shape: Tuple[int, ...]) -> Array:
    """
    Unpack matrix to kernel: (n_x*|L|, n_y*|R|) -> (n_x, n_y, *L, *R).
    
    Args:
        M: Packed matrix
        n_x: Number of x points
        n_y: Number of y points
        left_shape: Left event shape
        right_shape: Right event shape
        
    Returns:
        Unpacked kernel with event dimensions
    """
    return M.reshape(n_x, n_y, *left_shape, *right_shape)


def pack_vector(v: Array, out_shape: Tuple[int, ...]) -> Array:
    """
    Pack vector with event dims: (n, *out_shape) -> (n*|out_shape|, 1).
    
    Args:
        v: Input vector
        out_shape: Event shape
        
    Returns:
        Packed column vector
    """
    v = jnp.asarray(v)
    n = v.shape[0]
    return v.reshape(n * prod(out_shape), 1)


def pack_vector_flat(v: Array, out_shape: Tuple[int, ...]) -> Array:
    """
    Pack vector to 1D: (n, *out_shape) -> (n*|out_shape|,).
    
    Args:
        v: Input vector
        out_shape: Event shape
        
    Returns:
        Flattened 1D vector
    """
    v = jnp.asarray(v)
    n = v.shape[0]
    return v.reshape(n * prod(out_shape),)


def unpack_vector(v_flat: Array, n: int, out_shape: Tuple[int, ...]) -> Array:
    """
    Unpack flat vector: (n*|out_shape|,) -> (n, *out_shape).
    
    For scalar output (empty out_shape), returns (n,).
    
    Args:
        v_flat: Flattened vector
        n: Number of data points
        out_shape: Event shape
        
    Returns:
        Unpacked vector with event dimensions
    """
    v_flat = jnp.asarray(v_flat)
    if not out_shape:  # Scalar case
        return v_flat.reshape(n,)
    return v_flat.reshape((n, *out_shape))

def pack_kernel(K_raw: Array, out_shape: Tuple[int, ...]) -> Array:
    """(nF, nG, *out_shape, *out_shape) -> (nF*L, nG*L)."""
    nF, nG = K_raw.shape[:2]
    L = _prod(out_shape) if out_shape else 1
    return K_raw.reshape(nF * L, nG * L)

def _flatten_kernel(K_raw: Array) -> Tuple[Array, int, Tuple[int, ...]]:
    """(nF,nG,*out,*out) -> (nF*L, nG*L), L, out_shape"""
    nF, nG = K_raw.shape[:2]
    out_shape = tuple(K_raw.shape[2:])
    L = _prod(out_shape) if out_shape else 1
    return K_raw.reshape(nF * L, nG * L), L, out_shape


# ============================================================================
# Utility Functions
# ============================================================================

def as_2d_column(v: Array) -> Tuple[Array, bool]:
    """
    Convert vector to 2D column, tracking if it was squeezed.
    
    Args:
        v: Input vector
        
    Returns:
        (2D column vector, was_squeezed flag)
    """
    if v.ndim == 1:
        return v[:, None], True
    return v, False


def restore_from_2d(y: Array, was_squeezed: bool) -> Array:
    """
    Restore original shape after as_2d_column.
    
    Args:
        y: 2D array
        was_squeezed: Whether original was 1D
        
    Returns:
        Array with original dimensionality
    """
    return y.squeeze(-1) if was_squeezed else y


def _append(shape: Tuple[int, ...], *more: int) -> Tuple[int, ...]:
    return tuple(shape) + tuple(more)

def _append_full(shape: Tuple[int, ...], tail: Tuple[int, ...]) -> Tuple[int, ...]:
    return (*shape, *tail)

def _append_flat(shape: Tuple[int, ...], in_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    return (*shape, _prod(in_shape))


# ============================================================================
# Legacy Aliases (for backward compatibility)
# ============================================================================

# Keep original names as aliases
canonicalise_K_axes = canonicalise_kernel_axes
_ensure_n_by_d = ensure_2d_column
_ensure_2d_rows = ensure_2d_rows
_prod = prod
_as_2d = as_2d_column
_restore = restore_from_2d
_enforce_event_shape = enforce_kernel_shape
_flatten_for_solver = flatten_kernel_for_solver
pack_ev2mat = pack_kernel_to_matrix
unpack_mat2ev = unpack_matrix_to_kernel
pack_vec = pack_vector
pack_vec_flat = pack_vector_flat
unpack_vec = unpack_vector
_unflatten_var_diag = unpack_vector
