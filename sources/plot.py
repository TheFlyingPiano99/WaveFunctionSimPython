import numpy as np
import math
import matplotlib.pyplot as plt


def plot_probability_evolution(probability_evolutions, delta_t, index, show_fig=False):
    plt.grid(True)
    plt.xlabel("Elapsed time [ħ/E]")
    plt.ylabel("Probability")
    plt.title('Probability of particle being found in different regions')
    n = probability_evolutions[0][0].size
    x = np.linspace(start=0, stop=n * delta_t, dtype=None, num=n)
    plt.xlim(0, n * delta_t)
    for prob_data in probability_evolutions:
        plt.plot(x, prob_data[0], label=prob_data[1])
    plt.legend()
    plt.savefig(f'output/probability_evolution_{index:4d}.png')
    if show_fig:
        plt.show()
