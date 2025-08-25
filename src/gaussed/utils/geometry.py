import jax, jax.numpy as jnp
from jax import Array

def safe_norm(x: Array, axis: int = -1, eps: float = 0.0) -> Array:
    sq = jnp.sum(x * x, axis=axis)
    return jnp.where(sq == 0, 0.0, jnp.sqrt(sq + eps))

@jax.custom_jvp
def pairwise_r(x: Array, y: Array, eps: float = 0.0) -> Array:
    return safe_norm(x - y, axis=-1, eps=eps)

@pairwise_r.defjvp
def _pairwise_r_jvp(primals, tangents):
    x, y, eps = primals
    dx, dy, deps = tangents
    r = pairwise_r(x, y, eps)
    diff = x - y
    num = jnp.sum(diff * (dx - dy), axis=-1)
    grad = jnp.where(r == 0.0, 0.0, num / r)
    return r, grad

def unit_vec(x: Array, y: Array, eps: float = 0.0) -> Array:
    diff = x - y
    r = pairwise_r(x, y, eps)[..., None]
    return jnp.where(r == 0.0, jnp.zeros_like(diff), diff / r)

