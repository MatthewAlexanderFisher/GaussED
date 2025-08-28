from jax import Array
import jax.numpy as jnp

def move_front_axis(a: Array, dst: int) -> Array:
    """Move leading axis 0 to position `dst` (0 <= dst <= a.ndim-1). Used to transform TensorKernels"""
    nd = a.ndim
    assert 0 <= dst < nd
    perm = [*range(1, dst + 1), 0, *range(dst + 1, nd)]
    return jnp.transpose(a, perm)
