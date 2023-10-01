import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image


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
