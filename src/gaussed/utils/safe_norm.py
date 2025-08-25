import jax, jax.numpy as jnp

@jax.custom_vjp
def safe_norm(d, axis=-1):
    return jnp.linalg.norm(d, axis=axis)

def _norm_fwd(d, axis=-1):
    r = jnp.linalg.norm(d, axis=axis)
    return r, (d, r, axis)

def _norm_bwd(res, g):
    d, r, axis = res
    # grad wrt d is (g/r) * d, but set to 0 when r==0
    scale = jnp.where(r > 0, g / r, 0.0)
    grad = jnp.expand_dims(scale, axis) * d
    return (grad, None)  # None for axis (static)

safe_norm.defvjp(_norm_fwd, _norm_bwd)
