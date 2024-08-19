import argparse
import json
import pathlib

import numpy as np
import pyvista as pv
import scipy.spatial
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--point_size", type=float, default=0.1)
parser.add_argument("--animate", '-a', action='store_true')
args = parser.parse_args()

point_size = args.point_size
CELL_COLOR = [136, 212, 171]
RECEPTOR_COLOR = [255, 155, 133]

folders = list(pathlib.Path('outs').iterdir())
last_folder = sorted(folders, key=lambda path: path.stat().st_ctime)[-1]
files = last_folder.rglob("*.npz")
files = sorted(files, key=lambda x: int(x.stem.split("_")[1]))

# Load the parameters
with open(last_folder/'params.json', 'r') as f:
    params = json.load(f)
    a = params["a"]
    harm_order = params["harm_order"]
    harm_degree = params["harm_degree"]
    gradient = np.array(params["gradient"])

def to_cartesian(r, theta, phi):
    r = a + np.real(scipy.special.sph_harm(harm_order, harm_degree, phi, theta))
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.stack([x, y, z], axis=-1)

comparison_files = [files[0], files[-1]]

t, p = np.meshgrid(np.linspace(0, np.pi, 1000), np.linspace(0, 2*np.pi, 1000))
surface_points = to_cartesian(a, t, p).reshape(-1, 3)
surface_cloud = pv.PolyData(surface_points)
arrow = pv.Arrow(np.array([1.8*a, 0, 0]), np.array(gradient)/np.linalg.norm(gradient), scale=0.8)

pl = pv.Plotter(shape=(1,2), window_size=[1800, 800])
for i, file in enumerate(comparison_files):
    pl.subplot(0, i)
    pl.add_mesh(surface_cloud, color=CELL_COLOR)
    pl.add_mesh(arrow)
    data = np.load(file)
    points = np.array(to_cartesian(a, data["thetas"], data["phis"]))
    receptors = pv.PolyData(points)
    pl.add_mesh(receptors, render_points_as_spheres=True, point_size=point_size * 100, color=RECEPTOR_COLOR)
    pl.add_axes()
    pl.view_isometric()
pl.link_views()
pl.show()

if not args.animate:
    exit()

pl = pv.Plotter(window_size=[800, 800], off_screen=False)
pl.add_mesh(surface_cloud, color=CELL_COLOR)
pl.add_mesh(arrow)
data = np.load(files[0])
points = np.array(to_cartesian(a, data["thetas"], data["phis"]))
receptors = pv.PolyData(points)
pl.add_mesh(receptors, render_points_as_spheres=True, point_size=point_size * 100, color=RECEPTOR_COLOR)
pl.add_axes()
pl.isometric_view()
pl.open_gif(last_folder / "animation.gif")
for f in tqdm(files[1:], ncols=81):
    data = np.load(f)
    pl.title_text = f"Frame: {f.stem}"
    new_points = to_cartesian(a, data["thetas"], data["phis"])
    receptors.points = new_points
    pl.write_frame()
pl.close()

