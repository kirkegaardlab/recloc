"""
This is the most generic script on this repository.
It is used to optimize the position of the receptors on the cell surface
to estimate the concentration gradient.

This describes the base model, and does not include multiple runs to test parameters.

Note: JAX spherical harmonics implementation is not working. The hacky way here is to manually set
the spherical harmonics to the desired degree and order.

Author: Albert Alonso
Date: 2024-09-01
"""
from functools import partial
import json
import os
import pathlib

import jax
import jax.numpy as jnp
import jax.scipy.optimize
import numpy as np
import optax
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)

# Parameters
a = 1.0  # radius of the cell (um)
c0 = 500.0  # constant offset in concentration (molecules/um^3)
D = 100  # diffusion coefficient (um^2/s)
tau = 1e-6  # measurement time (s)
D_CORR = 4 * D * tau  # correlation distance (Diffusion * measurement time)
M = 500  # number of surface receptors (N)
GRADIENT = jnp.array([10.0, 0.0, 00.0])  # gradient vector (molecules/um^3/um)
MAX_ITER = 10_000  # number of maximum iterations
perturbation_strength = 1.0
harm_degree = 3  # degree of the spherical harmonic (l >= m)
harm_order = 2  # spherical harmonic order (m)
learning_rate = 1e-3  # learning rate for the optimizer


def fibonacci_sphere(n):
    """Place n receptors on the surface of a sphere."""
    phi = jnp.pi * (jnp.sqrt(5) - 1)
    idxs = jnp.arange(n)
    y = 1 - idxs / (n - 1) * 2
    radius = jnp.sqrt(1 - y * y)
    theta = phi * idxs
    x = jnp.cos(theta) * radius
    z = jnp.sin(theta) * radius
    theta = jnp.arccos(z)
    phi = jnp.arctan2(y, x)
    return theta, phi


def cramer_rao_bound(X, G):
    mu = X @ G  # X={x_0, ..., x_M}, x_i=(1, x, y, z), G=(c0, g_x, g_y, g_z)
    sigmas = jnp.sqrt(mu)
    scales = jnp.sum((X[:, None] - X[None, :]) ** 2, axis=-1) / (4 * D * tau)
    # We approximate the Eq.(3) with the following exponential since the 
    # exponential integral function yields nan gradients.
    # spatial_corr = exp(-λ) + λ expint(-λ) ≈ exp(-2λ)
    spatial_corr = jnp.exp(-2*scales)
    cov = (sigmas[:, None] + sigmas[None, :]) / 2 * spatial_corr
    cov_inv = jnp.linalg.inv(cov)
    cov_part_deriv = spatial_corr[...,None] / 2 * (X[:, None] + X[None, :])
    A = jnp.einsum("ij,jka->ika", cov_inv, cov_part_deriv)
    B = jnp.einsum("ija,jkb->ikab", A, A)
    I = X.T @ cov_inv @ X + 0.5 * jnp.trace(B)
    cov_lbound = jnp.linalg.inv(I)
    return cov_lbound


def gradient_estimation_covariances(X, g):
    X_hat = jnp.insert(X, 0, 1.0, axis=1)  # X=N * (x, y, z) -> X_hat=N * (1, x, y, z)
    G_hat = jnp.insert(g, 0, c0, axis=0)  # G=(gx, gy, gz) -> G_hat=(c0, gx, gy, gz)
    return cramer_rao_bound(X_hat, G_hat)


def to_cartesian(a, theta, phi, alpha=0.0):
    sphe = (harm_degree, harm_order)
    # r = a + jnp.real(jax.scipy.special.sph_harm(harm_order, harm_degree, phi, theta))
    if sphe == (2, 0):
        Y = 1 / 4 * jnp.sqrt(5 / jnp.pi) * (3 * jnp.cos(theta) ** 2 - 1.0)
    elif sphe == (2, 2):
        Y = 1 / 4 * jnp.sqrt(15 / (2 * jnp.pi)) * jnp.cos(2 * phi) * jnp.sin(theta) ** 2
    elif sphe == (3, 1):
        Y = (-1/8 * jnp.sqrt(21 / jnp.pi) * jnp.cos(phi) * jnp.sin(theta) * (5 * jnp.cos(theta) ** 2 - 1))
    elif sphe == (3, 2):
        Y = (1 / 4 * jnp.sqrt(105 / (jnp.pi * 2)) * jnp.cos(2 * phi) * jnp.sin(theta) ** 2 * jnp.cos(theta))
    elif sphe == (4, 2):
        Y = (3 / 8 * jnp.sqrt(5 / (2 * jnp.pi)) * jnp.cos(2 * phi) * jnp.sin(theta) ** 2 * (7 * jnp.cos(theta) ** 3 - 3 * jnp.cos(theta)))
    elif sphe == (5, 3):
        Y = 3/ 256 * jnp.sqrt(5005 / (2 * jnp.pi)) * jnp.cos(4 * phi) * jnp.sin(theta) ** 4 * (323 * jnp.cos(theta) ** 6 - 255 * jnp.cos(theta) ** 4 + 45 * jnp.cos(theta) ** 2 - 1)
    elif sphe == (6, 0):
        Y = ( 1 / 32 * jnp.sqrt(13 / jnp.pi) * ( 231 * jnp.cos(theta) ** 6 - 315 * jnp.cos(theta) ** 4 + 105 * jnp.cos(theta) ** 2 - 5))
    elif sphe == (6, 4):
        Y = 3 / 32 * jnp.sqrt(91 / (2 * jnp.pi)) * jnp.cos(4 * phi) * jnp.sin(theta) ** 4 * (11 * jnp.cos(theta) ** 2 - 1)
    elif sphe == (7, 4):
        Y = 3 / 32 * jnp.sqrt(385 / (2 * jnp.pi)) * jnp.cos(4 * phi) * jnp.sin(theta) ** 4 * (13 * jnp.cos(theta) ** 3 - 3 * jnp.cos(theta))
    else:
        Y = 0
    r = a * (1 + alpha * Y)
    x = r * jnp.sin(theta) * jnp.cos(phi)
    y = r * jnp.sin(theta) * jnp.sin(phi)
    z = r * jnp.cos(theta)
    return jnp.stack([x, y, z], axis=-1)


@jax.jit
@partial(jax.value_and_grad, argnums=(0, 1))
def objective_fun(thetas, phis):
    X = to_cartesian(a, thetas, phis, alpha=perturbation_strength)
    estimation_covariance = gradient_estimation_covariances(X, GRADIENT)
    return jnp.trace(estimation_covariance)


if __name__ == "__main__":

    thetas, phis = fibonacci_sphere(M)

    # Create the output folder (clean it if it exists)
    output_folder = pathlib.Path("outputs")
    if not output_folder.exists():
        os.system(f"rm -rf {output_folder}/*")
    ckpt_folder = output_folder / "checkpoints"
    ckpt_folder.mkdir(parents=True, exist_ok=True)
    params = {"harm_order": harm_order,"harm_degree": harm_degree}
    (output_folder / "params.json").write_text(json.dumps(params, indent=4))

    # Initialise the optimizer and begin the optimization process
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init((thetas, phis))
    rtol, atol, error_prev = 1e-9, 1e-10, jnp.inf
    metrics = []
    for i in (bar := tqdm(range(MAX_ITER), ncols=81)):
        error, grads = objective_fun(thetas, phis)

        if i == 0:
            tqdm.write(f"Initial error (N={len(thetas)}): {error:.5g}")
        bar.set_description(f"Loss: {error:.5g}")

        if jnp.isnan(error):
            tqdm.write("NaN detected. Exiting...")
            break

        if i % 10 == 0:
            jnp.savez(ckpt_folder / f"ckpt_{i:06d}.npz", thetas=thetas, phis=phis, error=error)

        update, opt_state = optimizer.update(grads, opt_state, (thetas, phis))
        (thetas, phis) = optax.apply_updates((thetas, phis), update)  # pyright: ignore
        thetas, phis = jnp.arccos(jnp.cos(thetas)), phis % (2 * jnp.pi) # pyright: ignore

        if jnp.abs(error - error_prev) < rtol * jnp.abs(error) + atol:
            jnp.savez(ckpt_folder / f"ckpt_{i:06d}.npz", thetas=thetas, phis=phis, error=error)
            break

        error_prev = error
        metrics.append(error)

    # Save the error and the final positions
    print(
        "Optimization finished with improvement:",
        metrics[-1] - metrics[0],
        metrics[-1] / metrics[0],
    )
    np.savetxt(output_folder / "metrics.txt", metrics)


