import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import json
import pathlib

import imageio.v2 as imagio
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

def to_cartesian(r, theta, phi, alpha=1):
    r = a + alpha * np.real(scipy.special.sph_harm(harm_order, harm_degree, phi, theta))
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.stack([x, y, z], axis=-1)

comparison_files = [files[0], files[-1]]

thetas = np.linspace(0, np.pi, 100)
phis = np.linspace(0, 2*np.pi, 100)
thetas, phis = np.meshgrid(thetas, phis)
arrow = pv.Arrow(np.array([1.8*a, 0, 0]), np.array(gradient)/np.linalg.norm(gradient), scale=0.8)

pl = pv.Plotter(shape=(1,2), window_size=[1800, 800])
for i, file in enumerate(comparison_files):
    pl.subplot(0, i)
    data = np.load(file)
    surface_points = to_cartesian(a, thetas, phis, alpha=data['alpha'])
    grid = pv.StructuredGrid(*surface_points.T)
    surface_mesh = grid.extract_geometry().clean()
    pl.add_mesh(surface_mesh, color=CELL_COLOR, smooth_shading=True, opacity=0.8)
    pl.add_mesh(arrow)
    points = np.array(to_cartesian(a, data["thetas"], data["phis"], data["alpha"]))
    receptors = pv.PolyData(points)
    pl.add_mesh(receptors, render_points_as_spheres=True, point_size=point_size * 100, color=RECEPTOR_COLOR)
    pl.add_axes()
    pl.view_isometric()
pl.link_views()
pl.show()

if not args.animate:
    exit()

frames_folder = last_folder / "frames"
frames_folder.mkdir(exist_ok=True)

def plot_cell(file):
    data = np.load(file)

    pl = pv.Plotter(window_size=[800, 800], off_screen=True)

    thetas = np.linspace(0, np.pi, 100)
    phis = np.linspace(0, 2*np.pi, 100)
    thetas, phis = np.meshgrid(thetas, phis)
    surface_points = to_cartesian(a, thetas, phis, alpha=data['alpha'])
    grid = pv.StructuredGrid(*surface_points.T)
    surface_mesh = grid.extract_geometry().clean()
    pl.add_mesh(surface_mesh, color=CELL_COLOR, opacity=0.7)

    arrow = pv.Arrow(np.array([1.8*a, 0, 0]), np.array(gradient)/np.linalg.norm(gradient), scale=0.8)
    pl.add_mesh(arrow)

    points = to_cartesian(a, data["thetas"], data["phis"], data["alpha"])
    receptors = pv.PolyData(points)
    pl.add_mesh(receptors, render_points_as_spheres=True, point_size=point_size * 100, color=RECEPTOR_COLOR)

    pl.add_text(f"Error: {data['error']:.5g}", position="upper_edge", font_size=12)

    pl.add_axes()
    pl.isometric_view()

    pl.screenshot(frames_folder/f'{file.stem}.png')
    pl.close()

if __name__ == "__main__":

    with ProcessPoolExecutor() as executor:
        tasks = [executor.submit(plot_cell, file) for file in files]
        for result in tqdm(as_completed(tasks), total=len(tasks), ncols=81):
            pass


    images = list(frames_folder.glob("*.png"))
    images = sorted(images, key=lambda x: int(x.stem.split("_")[1]))
    images = [imageio.imread(str(image)) for image in images]
    imageio.mimsave(last_folder / "animation.mp4", images, fps=30)


