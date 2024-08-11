import pathlib
import numpy as np
import pyvista as pv
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import ffmpeg

a = 1.0 # Sphere radius
point_size = 0.2 # Size of the receptors relative to the sphere

def to_cartesian(a, t, p):
    x = a * np.sin(t) * np.cos(p)
    y = a * np.sin(t) * np.sin(p)
    z = a * np.cos(t)
    return np.vstack([x, y, z]).T


# Sort the datafiles by the step
folders = list(pathlib.Path('outs').iterdir())
last_folder = sorted(folders, key=lambda path: path.stat().st_ctime)[-1]
files = last_folder.rglob("*.npz")
files = sorted(files, key=lambda x: int(x.stem.split("_")[1]))
frames_folder = last_folder / "frames"
frames_folder.mkdir(exist_ok=True)

def plot_sphere(file):
    pl = pv.Plotter(window_size=[800, 800], off_screen=True)
    data = np.load(file)
    t, p = data["thetas"], data["phis"]

    points = np.array(to_cartesian(a, t, p))
    sphere = pv.Sphere(radius=a, center=(0, 0, 0))
    pl.add_mesh(sphere, color=[136, 212, 171], smooth_shading=True)
    # Add receptor points on the sphere's surface
    receptors = pv.PolyData(points)
    pl.add_mesh(receptors, render_points_as_spheres=True, point_size=point_size * 100, color=[255, 155, 133])
    pl.add_axes()
    pl.isometric_view()
    pl.screenshot(frames_folder/f'{file.stem}.png')
    pl.close()

def main():
    with ProcessPoolExecutor() as executor:
        tasks = [executor.submit(plot_sphere, file) for file in files]
        for task in tqdm(as_completed(tasks), total=len(tasks), ncols=81):
            pass

    # Make a video (duration to be 10 seconds)
    num_frames = len(list(frames_folder.glob("*.png")))
    fps = max(int(num_frames // 10), 1)
    ffmpeg.input(
        str(frames_folder / "*.png"), pattern_type='glob', framerate=fps
    ).output(
        str(last_folder / "animation.mp4"), pix_fmt='yuv420p',
        vf='scale=trunc(iw/2)*2:trunc(ih/2)*2'
    ).run(quiet=True, overwrite_output=True)

if __name__ == "__main__":
    main()
