import numpy as np
import cupy as cp
import math
import sources.math_utils as math_utils
from numba import jit, njit


def P_free_space(r, t):
    return (
        1.0
        / (2.0 * math.pi * t) ** 0.5
        * math.exp(-1j * math.pi / 4)
        * math.exp(1j * cp.dot(r, r) / 2.0 / t)
    )


def wave_0_x(x):
    sum = cp.complex_(0.0)
    for i in range(10):
        sum += P_free_space(cp.array([x, 0]), i)
    return sum


def wave_0_y(y):
    sum = cp.complex_(0.0)
    for i in range(10):
        sum += P_free_space(cp.array([0, y]), i)
    return sum


def wave_packet(x, y):
    return wave_0_x(x) * wave_0_y(y)

@jit(nopython=True)
def init_gaussian_wave_packet(
    N: int,
    delta_x_bohr_radii: float,
    a: float,
    r_0_bohr_radii_3: np.array,
    initial_momentum_h_per_bohr_radius_3: np.array,
):
    wave_tensor = np.zeros(shape=(N, N, N), dtype=cp.complex_)
    for x in range(0, N):
        for y in range(0, N):
            for z in range(0, N):
                r = (
                    np.array([x, y, z]) * delta_x_bohr_radii
                    - np.array([1.0, 1.0, 1.0]) * N * delta_x_bohr_radii * 0.5
                )
                wave_tensor[x, y, z] = (
                    (2.0 / math.pi / a**2) ** (3.0 / 4.0)
                    * math_utils.exp_i(np.dot(initial_momentum_h_per_bohr_radius_3, r))
                    * math.exp(
                        -np.dot(r - r_0_bohr_radii_3, r - r_0_bohr_radii_3) / a**2
                    )
                )
    return wave_tensor
