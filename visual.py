import pathlib
import numpy as np
import pyvista as pv

a = 1.0

def to_cartesian(a, t, p):
    x = a * np.sin(t) * np.cos(p)
    y = a * np.sin(t) * np.sin(p)
    z = a * np.cos(t)
    return np.vstack([x, y, z]).T


# Sort the datafiles by the step
parent = pathlib.Path('outs/')
folders = list(parent.iterdir())
last_folder = sorted(folders, key=lambda x: int(x.stem.split("_")[0]))[-1]
files = last_folder.glob("*.npz")
files = sorted(files, key=lambda x: int(x.stem.split("_")[1]))

#files = ['initial.npz', 'directional.npz']
#files = [pathlib.Path(f) for f in files]


pl = pv.Plotter(shape=(1,2), window_size=[1800, 800])
for i, file in enumerate([files[0], files[-1]]):
    data = np.load(file)
    t, p = data["thetas"], data["phis"]

    pl.subplot(0, i)
    points = np.array(to_cartesian(a, t, p))
    pl.add_points(np.zeros((1, 3)), render_points_as_spheres=True, point_size=a * 430, color=[136, 212, 171])
    pl.add_points(points, render_points_as_spheres=True, point_size=a*20, color=[255, 155, 133])
    pl.add_axes()

pl.show()
#viewup = [0, 1, 0]
#path = pl.generate_orbital_path(factor=1.0, n_points=60, viewup=viewup)
#pl.open_gif("orbit.gif")
#pl.orbit_on_path(path, write_frames=True, viewup=viewup, step=1/30)
#pl.close()
