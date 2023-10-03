import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image
import sources.math_utils as math_utils
from PIL import Image


def plot_probability_evolution(probability_evolutions, delta_t, index, show_fig=False):
    plt.clf()  # Clear figure
    plt.grid(True)
    plt.xlabel("Elapsed time [ħ/E]")
    plt.ylabel("Probability")
    plt.title("Probability of particle being found in different regions")
    n = probability_evolutions[0][0].size
    x = np.linspace(start=0, stop=n * delta_t, dtype=None, num=n)
    plt.xlim(0, n * delta_t)
    for prob_data in probability_evolutions:
        plt.plot(x, prob_data[0], label=prob_data[1])
    plt.legend()
    plt.savefig(f"output/probability_evolution_{index:04d}.png")
    if show_fig:
        plt.show()


def plot_per_axis_probability_density(
    probability_densities, delta_x, delta_t, index, show_fig=False
):
    plt.clf()  # Clear figure
    plt.grid(True)
    plt.xlabel("Location [Bohr radius]")
    plt.ylabel("Probability density")
    plt.title(
        f"Probability density (Elapsed time = {index * delta_t:.5f} ħ/E = {math_utils.h_bar_per_hartree_to_ns(index * delta_t):.2E} ns)"
    )
    n = probability_densities[0][0].size
    # For n assuming that all datasets have the same size
    x = np.linspace(start=-n * delta_x * 0.5, stop=n * delta_x * 0.5, dtype=None, num=n)
    plt.xlim(-n * delta_x * 0.5, n * delta_x * 0.5)
    plt.ylim(0.0, 0.5)
    for prob_data in probability_densities:
        plt.plot(x, prob_data[0], label=prob_data[1])
    plt.legend()
    plt.savefig(f"output/per_axis_probability_density_{index:04d}.png")
    if show_fig:
        plt.show()
    fig = plt.gcf()
    fig.canvas.draw()
    rgb = fig.canvas.tostring_rgb()
    w, h = fig.canvas.get_width_height()
    img = Image.frombytes("RGB", (w, h), rgb)
    return np.array(img)


def plot_canvas(
    plane_probability_density,
    probability_save_path,
    plane_dwell_time_density,
    dwell_time_save_path,
):
    formatted = plane_probability_density
    matplotlib.image.imsave(
        fname=probability_save_path, arr=formatted, cmap="gist_heat", dpi=100
    )
    formatted = plane_dwell_time_density
    matplotlib.image.imsave(
        fname=dwell_time_save_path, arr=formatted, cmap="gist_heat", dpi=100
    )
