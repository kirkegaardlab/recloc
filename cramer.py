from datetime import datetime
from functools import partial
import json
import pathlib

import jax
import jax.numpy as jnp
import jax.scipy.optimize
import matplotlib.pyplot as plt
import optax
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)

a = 1.0  # radius of the cell
c0 = 100.0  # constant offset in concentration
D_CORR = 0.002  # correlation distance
M = 100  # number of surface receptors
GRADIENT = jnp.array([00.0, 00.0, 10.0])  # gradient vector
MAX_ITER = 100_000  # number of maximum iterations
harm_degree = [3]  # degree of the spherical harmonic (l >= m)
harm_order = 1  # spherical harmonic order (m)


def fibonacci_sphere(n):
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


def distance(X):
    r = jnp.linalg.norm(X, axis=1)
    cos_similarity = (X @ X.T) / (r[:, None] * r[None, :])
    cos_distance = 1 - cos_similarity
    return cos_distance


def cramer_rao_bound(X, G):
    mu = X @ G  # X={x_0, ..., x_M}, x_i=(1, x, y, z), G=(c0, g_x, g_y, g_z)
    sigma = jnp.sqrt(mu)
    spatial_corr = jnp.exp(-distance(X) / (2 * D_CORR**2))
    C = sigma[:, None] * sigma[None, :] * spatial_corr
    Cinv = jnp.linalg.inv(C)
    S = sigma[:, None] / sigma[None, :]

    @partial(jax.vmap, in_axes=(0, None, 0, 0))
    @partial(jax.vmap, in_axes=(None, 0, 0, 0))
    def partial_deriv(xi, xj, s, e):
        return e / 2 * (xi / s + xj * s)

    C_deriv = partial_deriv(X, X, S, spatial_corr)
    A = jnp.einsum("ij,jka->ika", Cinv, C_deriv)
    B = jnp.einsum("ija,jkb->ikab", A, A)
    I = X.T @ Cinv @ X + 0.5 * jnp.trace(B)
    cov_lbound = jnp.linalg.inv(I)
    return cov_lbound


def gradient_estimation_error(X, g):
    X_hat = jnp.insert(X, 0, 1.0, axis=1) # X=N * (x, y, z) -> X_hat=N * (1, x, y, z)
    G_hat = jnp.insert(g, 0, c0, axis=0) # G=(gx, gy, gz) -> G_hat=(c0, gx, gy, gz)
    covariance_estimations = cramer_rao_bound(X_hat, G_hat)
    variance_estimations = jnp.diag(covariance_estimations)
    return variance_estimations


def to_cartesian(a, theta, phi):
    r = a + jnp.real(jax.scipy.special.sph_harm(harm_order, harm_degree, phi, theta))
    x = r * jnp.sin(theta) * jnp.cos(phi)
    y = r * jnp.sin(theta) * jnp.sin(phi)
    z = r * jnp.cos(theta)
    return jnp.stack([x, y, z], axis=-1)


@jax.jit
@partial(jax.value_and_grad, argnums=(0, 1))
def objective_fn(thetas, phis):
    X = to_cartesian(a, thetas, phis)
    gradient_uncertainty = gradient_estimation_error(X, GRADIENT)
    error = gradient_uncertainty.sum()
    return error


def main():

    thetas, phis = fibonacci_sphere(M)

    output_folder = pathlib.Path("outs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_folder = output_folder / "checkpoints"
    ckpt_folder.mkdir(parents=True, exist_ok=True)
    params = {
        "a": a,
        "harm_order": harm_order,
        "harm_degree": harm_degree,
        "gradient": GRADIENT.tolist(),
    }
    (output_folder / "params.json").write_text(json.dumps(params, indent=4))

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init((thetas, phis))
    rtol, atol, error_prev = 1e-10, 1e-14, jnp.inf
    metrics = []
    for i in (bar := tqdm(range(MAX_ITER), ncols=81)):
        error, grads = objective_fn(thetas, phis)

        if i == 0:
            tqdm.write(f"Initial error: {error:.5g}")
        bar.set_description(f"Loss: {error:.5g}")

        if jnp.isnan(error):
            tqdm.write("NaN detected. Exiting...")
            break

        if i % 10 == 0:
            jnp.savez(ckpt_folder / f"ckpt_{i:06d}.npz", thetas=thetas, phis=phis)

        update, opt_state = optimizer.update(grads, opt_state, (thetas, phis))
        (thetas, phis) = optax.apply_updates((thetas, phis), update)  # pyright: ignore
        thetas = thetas % jnp.pi
        phis = phis % (2 * jnp.pi)

        if jnp.abs(error - error_prev) < rtol * jnp.abs(error) + atol:
            jnp.savez(ckpt_folder / f"ckpt_{i:06d}.npz", thetas=thetas, phis=phis)
            break

        error_prev = error
        metrics.append(error)

    plt.figure(dpi=300)
    plt.plot(metrics)
    plt.title("Gradient estimation error")
    plt.xlabel("Iteration")
    plt.ylabel("MSE")
    plt.tight_layout()
    plt.savefig(output_folder / "error.png")
    plt.show()


if __name__ == "__main__":
    main()
