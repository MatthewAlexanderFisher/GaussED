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

def _append(shape: Tuple[int, ...], *more: int) -> Tuple[int, ...]:
    return tuple(shape) + tuple(more)

def _append_full(shape: Tuple[int, ...], tail: Tuple[int, ...]) -> Tuple[int, ...]:
    return (*shape, *tail)

def _append_flat(shape: Tuple[int, ...], in_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    return (*shape, _prod(in_shape))


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

# Pack event dimensions into a matrix (used to coerce LinOp format)
def pack_ev2mat(K: Array,
                left_shape: Tuple[int, ...],
                right_shape: Tuple[int, ...]) -> Array:
    """(n_x, n_y, *L, *R) -> (n_x*|L|, n_y*|R|)"""
    n_x, n_y = K.shape[:2]
    return K.reshape(n_x * _prod(left_shape), n_y * _prod(right_shape))

def unpack_mat2ev(M: Array,
                  n_x: int, n_y: int,
                  left_shape: Tuple[int, ...],
                  right_shape: Tuple[int, ...]) -> Array:
    """(n_x*|L|, n_y*|R|) -> (n_x, n_y, *L, *R)"""
    return M.reshape(n_x, n_y, *left_shape, *right_shape)


def _as_2d(V: Array):
    return (V[:, None], True) if V.ndim == 1 else (V, False)

def _restore(U: Array, was_vec: bool):
    return U.squeeze(-1) if was_vec else U


def pack_vec(v_raw: Array, out_shape: Tuple[int, ...]) -> Array:
    """(n, *out_shape) -> (n*L, 1). Scalar-out -> (n,1)."""
    v_raw = jnp.asarray(v_raw)
    n = v_raw.shape[0]
    L = _prod(out_shape) if out_shape else 1
    return v_raw.reshape(n * L, 1)

def pack_vec_flat(v_raw: Array, out_shape: Tuple[int, ...]) -> Array:
    """(n, *out_shape) -> (n*L,). Keep only if you really need 1-D."""
    v_raw = jnp.asarray(v_raw)
    n = v_raw.shape[0]
    L = _prod(out_shape) if out_shape else 1
    return v_raw.reshape(n * L,)


def unpack_vec(v_flat: Array, n: int, out_shape: Tuple[int, ...]) -> Array:
    """(n*L,) -> (n, *out_shape). Scalar-out: (n,) stays (n,)."""
    v_flat = jnp.asarray(v_flat)
    if not out_shape:
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

def _unflatten_var_diag(v_flat: Array, nG: int, out_shape: Tuple[int, ...]) -> Array:
    """(nG*L,) -> (nG,*out) or (nG,) when scalar."""
    if not out_shape:
        return v_flat.reshape(nG,)
    return v_flat.reshape((nG, *out_shape))
