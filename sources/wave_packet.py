import numpy as np
import cupy as cp
import math
import sources.math_utils as math_utils
from numba import jit, njit
from pathlib import Path
import os

def P_free_space(r, t):
    return (
        1.0
        / (2.0 * math.pi * t) ** 0.5
        * math.exp(-1j * math.pi / 4)
        * math.exp(1j * cp.dot(r, r) / 2.0 / t)
    )


def wave_0_x(x):
    sum = cp.csingle(0.0)
    for i in range(10):
        sum += P_free_space(cp.array([x, 0]), i)
    return sum


def wave_0_y(y):
    sum = cp.csingle(0.0)
    for i in range(10):
        sum += P_free_space(cp.array([0, y]), i)
    return sum


def wave_packet(x, y):
    return wave_0_x(x) * wave_0_y(y)


wave_packet_kernel_source = Path("sources/cuda_kernels/gaussian_wave_packet.cu").read_text().replace("PATH_TO_SOURCES", os.path.abspath("sources"))

def init_gaussian_wave_packet(
    delta_x_bohr_radii_3: np.array,
    a: float,
    r_0_bohr_radii_3: np.array,
    initial_momentum_h_per_bohr_radius_3: np.array,
    shape: np.shape,
):
    wave_packet_kernel = cp.RawKernel(wave_packet_kernel_source,
                                 'wave_packet_kernel',
                                 enable_cooperative_groups=False)
    grid_size = math_utils.get_grid_size(shape)

    block_size = (shape[0] // grid_size[0], shape[1] // grid_size[1], shape[2] // grid_size[2])
    wave_tensor = cp.zeros(shape=shape, dtype=cp.csingle)
    wave_packet_kernel(
        grid_size,
        block_size,
        (
            wave_tensor,

            cp.float32(delta_x_bohr_radii_3[0]),
            cp.float32(delta_x_bohr_radii_3[1]),
            cp.float32(delta_x_bohr_radii_3[2]),

            cp.float32(a),

            cp.float32(r_0_bohr_radii_3[0]),
            cp.float32(r_0_bohr_radii_3[1]),
            cp.float32(r_0_bohr_radii_3[2]),

            cp.float32(initial_momentum_h_per_bohr_radius_3[0]),
            cp.float32(initial_momentum_h_per_bohr_radius_3[1]),
            cp.float32(initial_momentum_h_per_bohr_radius_3[2])
        )
    )
    return wave_tensor
