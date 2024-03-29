import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image
import sources.math_utils as math_utils
from matplotlib.image import NonUniformImage
from PIL import Image
import os

font_size = 16 # 18 would be too big

def plot_probability_evolution(out_dir, probability_evolutions, delta_t, index, show_fig=False):
    # Path:
    dir = os.path.join(out_dir, "probability_evolution/")
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
    matplotlib.rcParams.update({'font.size': font_size})
    plt.clf()  # Clear figure
    plt.grid(True)
    plt.xlabel("Elapsed time [ħ/Hartree]")
    plt.ylabel("Probability")
    plt.title("Integrated probability density")
    n = probability_evolutions[0][0].size # Assuming that all lists are of the same size
    x = np.linspace(start=0, stop=n * delta_t, dtype=None, num=n)
    plt.xlim(0, n * delta_t)
    plt.ylim(0.0, 2.0)
    for prob_data in probability_evolutions:
        plt.plot(x, prob_data[0], label=prob_data[1])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(dir, f"probability_evolution_{index:04d}.png"))
    if show_fig:
        plt.show()


def plot_per_axis_probability_density(
    out_dir, title: str, data: tuple, delta_x: float, delta_t: float, index: int, potential_scale: float, show_fig=False
):
    matplotlib.rcParams.update({'font.size': font_size})
    plt.clf()  # Clear figure

    dir = os.path.join(out_dir, "per_axis_probability_density/")
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)

    plt.grid(True)
    plt.xlabel("Location [Bohr radius]")
    plt.ylabel(f"Probability density / Potential [{1.0 / potential_scale:.1f} Hartree]", fontsize=font_size * 0.9)
    plt.title(f"Elapsed time = {index * delta_t:.2f} ħ/Hartree = {math_utils.h_bar_per_hartree_to_fs(index * delta_t):.2f} fs")
    n = data[0][0].size
    # For n assuming that all datasets have the same size
    x = np.linspace(start=-n * delta_x * 0.5, stop=n * delta_x * 0.5, dtype=None, num=n)
    plt.xlim(data[0][2], data[0][3])
    plt.ylim(0.0, 0.25)
    for prob_data in data:
        plt.plot(x, prob_data[0], label=prob_data[1])
    plt.legend()
    plt.subplots_adjust(left=0.14, bottom=0.12, right=0.95, top=0.9)
    plt.savefig(
        os.path.join(
            dir,
            f"per_axis_probability_density_{index:04d}.png",
        )
    )
    if show_fig:
        plt.show()
    fig = plt.gcf()
    fig.canvas.draw()
    rgb = fig.canvas.tostring_rgb()
    w, h = fig.canvas.get_width_height()
    img = Image.frombytes("RGB", (w, h), rgb)
    return np.array(img)


def plot_canvas(out_dir, plane_probability_density, plane_dwell_time_density, index, delta_x, delta_t):
    # Probability density:
    dir = os.path.join(out_dir, "canvas_probability/")
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)

    plt.clf()

    matplotlib.rcParams.update({'font.size': 14})
    plt.imshow(plane_probability_density,
               cmap="Reds",
               interpolation="bilinear",
               vmin=0.0,
               vmax=max(0.0000000001, np.max(plane_probability_density)),
               )
    plt.colorbar()
    plt.xlabel("X coordinate [Bohr radius]")
    plt.ylabel("Y coordinate [Bohr radius]")
    plt.xlim(0, plane_probability_density.shape[0])
    plt.ylim(0, plane_probability_density.shape[1])
    plt.xticks(ticks=np.linspace(0, plane_probability_density.shape[0], 5),
               labels=np.linspace(-plane_probability_density.shape[0] * delta_x * 0.5, plane_probability_density.shape[0] * delta_x * 0.5, 5))
    plt.yticks(ticks=np.linspace(0, plane_probability_density.shape[1], 5),
               labels=np.linspace(-plane_probability_density.shape[1] * delta_x * 0.5, plane_probability_density.shape[1] * delta_x * 0.5, 5))
    plt.title(f"Elapsed time = {index * delta_t:.2f} ħ/Hartree = {math_utils.h_bar_per_hartree_to_fs(index * delta_t):.2f} fs\n ")
    plt.tight_layout()
    plt.savefig(fname=os.path.join(dir, f"measurement_plane_probability_{index:04d}.png"))

    # Dwell time:
    dir = os.path.join(out_dir, "canvas_dwell_time/")
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)

    plt.clf()
    plt.imshow(plane_dwell_time_density,
               cmap="Reds",
               interpolation="bilinear",
               vmin=0.0,
               vmax=max(0.0000000001, np.max(plane_dwell_time_density)),
               )
    plt.colorbar()
    plt.xlabel("X coordinate [Bohr radius]")
    plt.ylabel("Y coordinate [Bohr radius]")
    plt.xlim(0, plane_dwell_time_density.shape[0])
    plt.ylim(0, plane_dwell_time_density.shape[1])
    plt.xticks(ticks=np.linspace(0, plane_dwell_time_density.shape[0], 5),
               labels=np.linspace(-plane_dwell_time_density.shape[0] * delta_x * 0.5, plane_dwell_time_density.shape[0] * delta_x * 0.5, 5))
    plt.yticks(ticks=np.linspace(0, plane_probability_density.shape[1], 5),
               labels=np.linspace(-plane_dwell_time_density.shape[1] * delta_x * 0.5, plane_dwell_time_density.shape[1] * delta_x * 0.5, 5))
    plt.title(f"Elapsed time = {index * delta_t:.2f} ħ/Hartree = {math_utils.h_bar_per_hartree_to_fs(index * delta_t):.2f} fs\n ")
    plt.tight_layout()
    plt.savefig(fname=os.path.join(dir, f"measurement_plane_dwell_time_{index:04d}.png"))
