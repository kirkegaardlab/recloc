from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.optimize
import matplotlib.pyplot as plt
import optax
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)

a = 1.0 # radius of the cell
c0 = 100.0 # constant offset in concentration
d_corr = 0.01 # correlation distance
N = 20_000 # number of realisations
m = 100 # number of surface receptors
GRADIENT = jnp.array([0.0, 0.0, 0.0]) # gradient vector
num_iters = 4_000


def expected_measurements(x, g):
    return c0 + x @ g


def cosine_distance(A, B):
    cos_similarity = (A @ B.T) / (jnp.linalg.norm(A, axis=1) * jnp.linalg.norm(B, axis=1))
    cos_distance = 1 - cos_similarity
    return cos_distance


def covariance_measurements(X, variance, d_corr=d_corr):
    D = cosine_distance(X, X)
    C = jnp.sqrt(variance[:, None] * variance[None, :]) * jnp.exp(-D / (2 * d_corr**2))
    return C


def solve_gradient(sample, X, C):
    C_inv = jnp.linalg.inv(C)
    A = X.T @ C_inv @ X
    b = X.T @ C_inv @ (sample - c0)
    return jnp.linalg.solve(A, b)


def to_cartesian(r, theta, phi):
    x = r * jnp.sin(theta) * jnp.cos(phi)
    y = r * jnp.sin(theta) * jnp.sin(phi)
    z = r * jnp.cos(theta)
    return jnp.stack([x, y, z], axis=1)


def gradient_estimation_error(key, X, g):
    means = expected_measurements(X, g)
    Cs = covariance_measurements(X, means) + 1e0 * jnp.eye(m)
    measurements = jax.random.multivariate_normal(key, means, Cs, shape=())
    estimates = solve_gradient(measurements, X, Cs)
    return estimates

@jax.jit
@partial(jax.value_and_grad, has_aux=True, argnums=(0, 1))
def objective_fn(thetas, phis, key, sigma=0.1):
    X = to_cartesian(a, thetas, phis)
    keys = jax.random.split(key, N)
    g_reals = jax.random.normal(key, shape=(N, 3)) * sigma + GRADIENT
    g_estimates = jax.vmap(gradient_estimation_error, in_axes=(0, None, 0))(keys, X, g_reals)
    error = jnp.mean((g_estimates - g_reals) ** 2, axis=0).mean()
    return error, (g_estimates, g_reals)

def main():
    okey = jax.random.key(42)
    thetas, phis = jax.random.uniform(okey, (2, m), minval=0, maxval=2 * jnp.pi)
    jnp.savez("initial.npz", thetas=thetas, phis=phis)

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init((thetas, phis))
    rtol, atol, error_prev = 1e-8, 1e-10, jnp.inf
    metrics = []
    for i in (bar := tqdm(range(num_iters), ncols=81)):
        key = jax.random.fold_in(okey, i)
        (error, _), grads = objective_fn(thetas, phis, key, sigma=10.0)

        update, opt_state = optimizer.update(grads, opt_state)
        if i == 0:
            tqdm.write(f"Initial error: {error:.5g}")
        bar.set_description(f"Loss: {error:.5g}")

        if jnp.isnan(error):
            break

        (thetas, phis) = optax.apply_updates((thetas, phis), update) # pyright: ignore
        thetas = thetas % (jnp.pi) # pyright: ignore
        phis = phis % (2*jnp.pi) # pyright: ignore

        if jnp.abs(error - error_prev) < rtol * jnp.abs(error) + atol:
            break
        error_prev = error

        metrics.append(error)
    jnp.savez("final.npz", thetas=thetas, phis=phis)

    plt.figure(dpi=300)
    plt.plot(metrics)
    plt.title("Gradient estimation error")
    plt.xlabel("Iteration")
    plt.ylabel("MSE")
    plt.tight_layout()
    plt.savefig('error.png')
    plt.show()

if __name__ == "__main__":
    main()
