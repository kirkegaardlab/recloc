from datetime import datetime
from functools import partial
import pathlib
import jax
import jax.numpy as jnp
import jax.scipy.optimize
import matplotlib.pyplot as plt
import optax
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)
# jax.disable_jit()

a = 1.0  # radius of the cell
c0 = 15.0  # constant offset in concentration
d_corr = 1e-3  # correlation distance
m = 200  # number of surface receptors
GRADIENT = jnp.array([10.0, 0.0, 0.0])  # gradient vector
num_iters = 10_000 # number of iterations


def fibonacci_sphere(m):
    phi = jnp.pi * (jnp.sqrt(5) - 1)
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


def covariance_measurements_derivative(X, variance, d_corr=d_corr):
    D = cosine_distance(X, X)
    exp_term = jnp.exp(-D / (2 * d_corr**2))

    sqrt_term = 1 / (2 * jnp.sqrt(variance[:, None] * variance[None, :]))

    C_c0 = (2 * c0 + (X @ GRADIENT)[:, None] + (X @ GRADIENT)[None, :])
    C_gx = (X[:, 0][:, None] * (c0 + X @ GRADIENT)[None, :] 
                      + X[:, 0][None, :] * (c0 + X @ GRADIENT)[:, None])
    C_gy = (X[:, 1][:, None] * (c0 + X @ GRADIENT)[None, :] 
                      + X[:, 1][None, :] * (c0 + X @ GRADIENT)[:, None])
    C_gz = (X[:, 2][:, None] * (c0 + X @ GRADIENT)[None, :] 
                      + X[:, 2][None, :] * (c0 + X @ GRADIENT)[:, None])

    C_deriv = jnp.stack((C_c0, C_gx, C_gy, C_gz), axis=-1)

    return C_deriv * (exp_term * sqrt_term)[..., None]



def to_cartesian(r, theta, phi):
    x = r * jnp.sin(theta) * jnp.cos(phi)
    y = r * jnp.sin(theta) * jnp.sin(phi)
    z = r * jnp.cos(theta)
    return jnp.stack([x, y, z], axis=1)


def cramer_rao_bound(X, means, C):
    Cinv = jnp.linalg.inv(C)
    I = X.T @ Cinv @ X
    
    C_deriv = covariance_measurements_derivative(X[:, -3:], means)
    A = jnp.einsum('ij,jka->ika', Cinv, C_deriv)
    B = jnp.einsum('ija,jkb->ikab', A, A)

    I = I + 0.5 * jnp.trace(B)

    uncertainty = jnp.linalg.inv(I)
    return uncertainty


def gradient_estimation_error(X, g):
    X_hat = jnp.pad(X, ((0, 0), (1, 0)), constant_values=1)
    means = expected_measurements(X, g)
    C = covariance_measurements(X, means)
    covariance_estimations = cramer_rao_bound(X_hat, means, C)

    variance_estimations = jnp.diag(covariance_estimations)[-3:]
    return variance_estimations


@jax.jit
@partial(jax.value_and_grad, argnums=(0,1))
def objective_fn(thetas, phis):
    X = to_cartesian(a, thetas, phis)
    gradient_uncertainty = gradient_estimation_error(X, GRADIENT)
    error = gradient_uncertainty.sum()
    return error


def main():

    thetas, phis = fibonacci_sphere(m)

    output_folder = pathlib.Path('outs') / datetime.now().strftime('%Y%m%d_%H%M%S')
    ckpt_folder = output_folder / 'checkpoints'
    ckpt_folder.mkdir(parents=True, exist_ok=True)

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init((thetas, phis))
    rtol, atol, error_prev = 1e-10, 1e-14, jnp.inf
    metrics = []
    for i in (bar := tqdm(range(num_iters), ncols=81)):
        error, grads = objective_fn(thetas, phis)

        # exit()


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
