import pathlib
import numpy as np
import pyvista as pv

a = 1.0 # Sphere radius
point_size = 0.2 # Size of the receptors relative to the sphere

def to_cartesian(a, t, p):
    x = a * np.sin(t) * np.cos(p)
    y = a * np.sin(t) * np.sin(p)
    z = a * np.cos(t)
    return np.vstack([x, y, z]).T


folders = list(pathlib.Path('outs').iterdir())
last_folder = sorted(folders, key=lambda path: path.stat().st_ctime)[-1]
files = last_folder.rglob("*.npz")
files = sorted(files, key=lambda x: int(x.stem.split("_")[1]))

pl = pv.Plotter(shape=(1,2), window_size=[1800, 800])
for i, file in enumerate([files[0], files[-1]]):
    data = np.load(file)
    t, p = data["thetas"], data["phis"]

    pl.subplot(0, i)
    points = np.array(to_cartesian(a, t, p))

    # Add a sphere representing the cell
    sphere = pv.Sphere(radius=a, center=(0, 0, 0))
    pl.add_mesh(sphere, color=[136, 212, 171], smooth_shading=True)

    # Add receptor points on the sphere's surface
    receptors = pv.PolyData(points)
    pl.add_mesh(receptors, render_points_as_spheres=True, point_size=point_size * 100, color=[255, 155, 133])
    pl.add_axes()
    pl.view_isometric()

pl.link_views()
pl.show()
