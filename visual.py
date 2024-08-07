import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from julius import to_cartesian, a



data = np.load("initial.npz")
initial = [data["thetas"], data["phis"]]


data  = np.load("final.npz")
final = [data["thetas"], data["phis"]]


fig = plt.figure(dpi=300)
for i, (t, p) in enumerate([initial, final]):
    ax = plt.subplot(1, 2, i + 1, projection="3d")
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 2000), np.linspace(0, np.pi, 2000))
    cx = a * np.cos(u) * np.sin(v)
    cy = a * np.sin(u) * np.sin(v)
    cz = a * np.cos(v)

    X = to_cartesian(a*1.05, t, p)

    # Flatten the grid arrays
    # Create grid points for the surface
    cx_flat = cx.flatten()
    cy_flat = cy.flatten()
    cz_flat = cz.flatten()
    grid_points = np.vstack([cx_flat, cy_flat, cz_flat])

    # Use kernel density estimation to find density on the surface
#    kde = gaussian_kde(X.T)
#    density = kde(grid_points).reshape(cx.shape)
#    density_normalized = (density - np.min(density)) / (np.max(density) - np.min(density))


    # Plot the surface with the density colormap
    #ax.plot_surface(cx, cy, cz, facecolors=plt.cm.Oranges(density_normalized))
    ax.plot_surface(cx, cy, cz, alpha=1.0, color="gray", zorder=2)
    ax.scatter(*X.T, color="blue", s=10, alpha=1.0, zorder=10)

    l = 1.1 * a
    ax.set(aspect="equal", xlim=(-l, l), ylim=(-l, l), zlim=(-l, l))
    ax.set_axis_off()

plt.show()
