from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.optimize
import matplotlib.pyplot as plt
import pathlib
import optax
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)

a = 1.0 # radius of the cell
c0 = 20.0 # constant offset in concentration
s = 1e-4
d_corr = 3 * s
m = 1000 # number of surface receptors
GRADIENT = jnp.array([10.0, 0.0, 0.0]) # gradient vector
num_iters = 10_000


def fibonacci_sphere(m):
    phi = jnp.pi * (jnp.sqrt(5.) - 1.)
    idxs = jnp.arange(m)
    y = 1 - idxs / (m - 1) * 2
    radius = jnp.sqrt(1 - y * y)
    theta = phi * idxs

    x = jnp.cos(theta) * radius
    z = jnp.sin(theta) * radius

    theta = jnp.arccos(z)
    phi = jnp.arctan2(y, x)
    return theta, phi


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

def to_cartesian(r, theta, phi):
    x = r * jnp.sin(theta) * jnp.cos(phi)
    y = r * jnp.sin(theta) * jnp.sin(phi)
    z = r * jnp.cos(theta)
    return jnp.stack([x, y, z], axis=1)


def cramer_rao_bound(X, _, C):
    I = X.T @ jnp.linalg.inv(C) @ X
    uncertainty = jnp.linalg.inv(I)
    return uncertainty


def gradient_estimation_error(_, X, g):
    X_hat = jnp.pad(X, ((0,0), (1,0)), constant_values=1)
    means = expected_measurements(X, g)
    C = covariance_measurements(X, means)
    variance_estimation = cramer_rao_bound(X_hat, means, C)
    variance_estimation = jnp.diag(variance_estimation)[-3:]
    return variance_estimation


@jax.jit
@partial(jax.value_and_grad, has_aux=True, argnums=(0, 1))
def objective_fn(thetas, phis, key):
    X = to_cartesian(a, thetas, phis)
    gradient_uncertainty = gradient_estimation_error(key, X, GRADIENT)
    error = jnp.mean(gradient_uncertainty)
    return error, gradient_uncertainty


def main():
    okey = jax.random.key(42)

    thetas, phis = fibonacci_sphere(m)

    output_folder = pathlib.Path('outs')
    output_folder.mkdir(exist_ok=True)
    output_folder = output_folder / str(len(list(output_folder.iterdir())))
    output_folder.mkdir(exist_ok=True)

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init((thetas, phis))
    rtol, atol, error_prev = 1e-10, 1e-14, jnp.inf
    metrics = []
    for i in (bar := tqdm(range(num_iters), ncols=81)):
        key = jax.random.fold_in(okey, i)
        (error, _), grads = objective_fn(thetas, phis, key)

        update, opt_state = optimizer.update(grads, opt_state)
        if i == 0:
            tqdm.write(f"Initial error: {error:.5g}")
        bar.set_description(f"Loss: {error:.5g}")

        if jnp.isnan(error):
            break

        if i % 100 == 0:
            jnp.savez(output_folder/f"ckpt_{i:06d}.npz", thetas=thetas, phis=phis)

        (thetas, phis) = optax.apply_updates((thetas, phis), update) # pyright: ignore

        if jnp.abs(error - error_prev) < rtol * jnp.abs(error) + atol:
            jnp.savez(output_folder/f"ckpt_{i:06d}.npz", thetas=thetas, phis=phis)
            break

        error_prev = error

        metrics.append(error)

    plt.figure(dpi=300)
    plt.plot(metrics)
    plt.title("Gradient estimation error")
    plt.xlabel("Iteration")
    plt.ylabel("MSE")
    plt.tight_layout()
    plt.savefig(output_folder/'error.png')
    plt.show()

if __name__ == "__main__":
    main()
