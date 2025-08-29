from jax import Array
import jax.numpy as jnp
from typing import Tuple

def move_front_axis(a: Array, dst: int) -> Array:
    """Move leading axis 0 to position `dst` (0 <= dst <= a.ndim-1). Used to transform TensorKernels"""
    nd = a.ndim
    assert 0 <= dst < nd
    perm = [*range(1, dst + 1), 0, *range(dst + 1, nd)]
    return jnp.transpose(a, perm)


def _ensure_n_by_d(x: Array) -> Array:
    x = jnp.asarray(x)
    return x.reshape(-1, 1) if x.ndim == 1 else x  # (n,) -> (n,1)

def _prod(shape: Tuple[int, ...]) -> int:
    return int(jnp.prod(jnp.array(shape))) if shape else 1


def ensure_n_by_shape(X: Array, item_shape: Tuple[int, ...]) -> Array:
    """Coerce X to (n, *item_shape). Accepts (n, *item_shape) or (*item_shape,) (n=1)."""
    X = jnp.asarray(X)
    if X.ndim == len(item_shape):         # e.g. (*item_shape,) -> (1,*item_shape)
        return X.reshape((1, *item_shape))
    if X.ndim == 1 and item_shape == (1,):  # common scalar case
        return X.reshape((-1, 1))
    # allow already-correct (n,*item_shape)
    if X.ndim == 1 + len(item_shape) and X.shape[1:] == item_shape:
        return X
    raise ValueError(f"Expected (n,{item_shape}) got {X.shape}")


def _enforce_event_shape(
    K: Array,
    n_x: int,
    n_y: int,
    left_shape: Tuple[int, ...],
    right_shape: Tuple[int, ...],
) -> Array:
    """Return K with shape (n_x, n_y, *left_shape, *right_shape)."""
    K = jnp.asarray(K)
    want_nd = 2 + len(left_shape) + len(right_shape)

    if K.ndim == want_nd:
        # Best case: already has the exact shape.
        return K

    # Common cases to fix up:

    # (i) Scalar output batched correctly: (n_x, n_y) with empty event shapes.
    if K.ndim == 2 and len(left_shape) == 0 and len(right_shape) == 0:
        return K

    # (ii) Scalar pair result for a single (x,y): () → (1,1)
    if K.ndim == 0 and n_x == 1 and n_y == 1 and len(left_shape) == 0 and len(right_shape) == 0:
        return K.reshape(1, 1)

    # (iii) Single (x,y) tensor-valued result: (*L, *R) → (1,1,*L,*R)
    if K.shape == (*left_shape, *right_shape) and n_x == 1 and n_y == 1:
        return K.reshape((1, 1, *left_shape, *right_shape))

    # (iv) Pairwise batched but one of the event dims squeezed accidentally:
    # Try to reshape if total size matches.
    size_ok = K.size == (n_x * n_y * (jnp.prod(jnp.array(left_shape)) or 1) * (jnp.prod(jnp.array(right_shape)) or 1))
    if size_ok:
        return K.reshape((n_x, n_y, *left_shape, *right_shape))

    # If we’re here, something is wrong upstream (wrong batching or wrong event dims).
    raise ValueError(
        f"KernelSpec: got K.shape={K.shape}, expected "
        f"(n_x, n_y, *left_shape, *right_shape)=({n_x},{n_y},{left_shape},{right_shape})"
    )


def _flatten_for_solver(K: Array) -> Array:
    # K: (n_x, n_y, *L, *R)  or  (n_x, n_y)
    if K.ndim == 2:  # scalar-output
        return K
    n_x, n_y = K.shape[:2]
    ex = K.shape[2:2 + (K.ndim - 2)//2]
    ey = K.shape[2 + (K.ndim - 2)//2:]
    K4 = K.reshape(n_x, n_y, int(jnp.prod(jnp.array(ex) or jnp.array([1]))),
                            int(jnp.prod(jnp.array(ey) or jnp.array([1]))))
    K4 = jnp.transpose(K4, (0, 2, 1, 3))
    return K4.reshape(n_x * K4.shape[1], n_y * K4.shape[3])
