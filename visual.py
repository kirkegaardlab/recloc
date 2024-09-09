from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import pathlib

import imageio.v2 as imageio
import numpy as np
import pyvista as pv
import scipy.spatial
from tqdm import tqdm

CELL_COLOR = (172,188,189)
RECEPTOR_COLOR = [255, 155, 133]

if __name__ == "__main__":

    ckpt_folder = pathlib.Path('outputs/checkpoints')
    if not ckpt_folder.exists():
        raise FileNotFoundError("No checkpoints found")

    ckpts = list(ckpt_folder.glob("*.npz"))
    ckpts = sorted(ckpts)
    selected_ckpts = [ckpts[0], ckpts[-1]]

    with open(ckpt_folder.parent/'params.json', 'r') as f:
        params = json.load(f)
        harm_order = params["harm_order"]
        harm_degree = params["harm_degree"]

    def to_cartesian(theta, phi, alpha=1):
        Y = np.real(scipy.special.sph_harm(harm_order, harm_degree, phi, theta))
        r = 1 + alpha * Y
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return np.stack([x, y, z], axis=-1)

    thetas = np.linspace(0, np.pi, 500)
    phis = np.linspace(0, 2*np.pi, 500)
    thetas, phis = np.meshgrid(thetas, phis)

    labels = ['initial', 'final']
    pl = pv.Plotter(shape=(1,2), window_size=[1600, 800], border=False)
    for i, file in enumerate(selected_ckpts):
        pl.subplot(0, i)
        pl.add_text(labels[i], position="upper_left", font_size=18, color='black')
        data = np.load(file)
        surface_points = to_cartesian(thetas, phis)
        grid = pv.StructuredGrid(*surface_points.T)
        surface_mesh = grid.extract_geometry().clean()
        pl.add_mesh(surface_mesh, smooth_shading=True, opacity=1.0, color=CELL_COLOR)
        points = np.array(to_cartesian(data["thetas"], data["phis"]))
        receptors = pv.PolyData(points)
        pl.add_mesh(receptors, render_points_as_spheres=True, point_size=15, color=RECEPTOR_COLOR)
        if i ==0:
            pl.add_axes(color='black', line_width=4)
        pl.view_isometric()
        pl.camera.zoom(1.28)
    pl.link_views()
    pl.show()

    print('Starting animation...')

    frames_folder = ckpt_folder.parent / "frames"
    frames_folder.mkdir(exist_ok=True)

    def plot_cell(file):
        data = np.load(file)
        pl = pv.Plotter(window_size=[800, 800], off_screen=True)
        thetas = np.linspace(0, np.pi, 400)
        phis = np.linspace(0, 2*np.pi, 400)
        thetas, phis = np.meshgrid(thetas, phis)
        surface_points = to_cartesian(thetas, phis)
        grid = pv.StructuredGrid(*surface_points.T)
        surface_mesh = grid.extract_geometry()
        pl.add_mesh(surface_mesh, opacity=0.7)
        points = to_cartesian(data["thetas"], data["phis"])
        receptors = pv.PolyData(points)
        pl.add_mesh(receptors, render_points_as_spheres=True, point_size=10, color=RECEPTOR_COLOR)
        pl.add_text(f"Error: {data['error']:.5g}", position="upper_edge", font_size=12)
        pl.add_axes()
        pl.isometric_view()
        pl.camera.zoom(1.2)
        pl.screenshot(frames_folder/f'{file.stem}.png')
        pl.close()


    #for file in tqdm(ckpts, ncols=81):
        #plot_cell(file)
    with ProcessPoolExecutor() as executor:
        tasks = [executor.submit(plot_cell, file) for file in ckpts]
        for result in tqdm(as_completed(tasks), total=len(tasks), ncols=81):
            pass

    images = list(frames_folder.glob("*.png"))
    images = sorted(images, key=lambda x: int(x.stem.split("_")[1]))
    images = [imageio.imread(str(image)) for image in images]
    imageio.mimsave(ckpt_folder.parent / "animation.mp4", images, fps=30)
    print('Animation saved to', ckpt_folder.parent / "animation.mp4")
