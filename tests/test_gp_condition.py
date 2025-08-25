import jax.numpy as jnp

from gaussed.gp.base import GP
from gaussed.domains.euclidean import Euclidean
from gaussed.codomains.base import Codomain
from gaussed.gp.kernels.rbf import RBF, RBFParams
from gaussed.engines.backends.cholesky import CholeskyBackend
from gaussed.engines.likelihood.gaussian_noise import Noiseless, HomoskedasticNoise


def build_gp():
    domain = Euclidean((1,))
    codomain = Codomain(())
    mean = lambda X: jnp.zeros(X.shape[0])
    params = RBFParams(raw_ell=jnp.array(0.0), raw_sigma2=jnp.array(0.0))
    kernel = RBF(params)
    return GP(domain, codomain, mean, kernel)


def test_condition_noiseless_matches_observations():
    gp = build_gp()
    X = jnp.array([[-1.0], [0.0], [1.0]])
    y = jnp.array([1.0, 0.0, -1.0])

    backend = CholeskyBackend()
    noise = Noiseless()
    post = gp.condition(X, y, backend=backend, noise=noise)

    assert jnp.allclose(post.mean(X), y, atol=1e-5)


def test_condition_homoskedastic_agrees_with_linear_system():
    gp = build_gp()
    X = jnp.array([[-1.0], [0.5]])
    y = jnp.array([1.0, -0.5])

    backend = CholeskyBackend()
    noise = HomoskedasticNoise(jnp.array(0.0))
    post = gp.condition(X, y, backend=backend, noise=noise)

    K_FF = gp.K(X, X)
    mu_F = jnp.zeros(X.shape[0])
    sigma2 = noise.sigma2()
    expected = mu_F + K_FF @ jnp.linalg.solve(K_FF + sigma2 * jnp.eye(X.shape[0]), y - mu_F)

    assert jnp.allclose(post.mean(X), expected, atol=1e-5)
    