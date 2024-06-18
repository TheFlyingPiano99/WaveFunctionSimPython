from numba import jit, vectorize
import numpy as np
import cupy as cp
import sources.math_utils as math_utils
from pathlib import Path
import os


def init_kinetic_operator(delta_x_3: np.array, delta_time: float, mass: float, shape: np.shape):
    grid_size = math_utils.get_grid_size(shape)
    block_size = (shape[0] // grid_size[0], shape[1] // grid_size[1], shape[2] // grid_size[2])

    f_x = cp.fft.fftfreq(n=shape[0], d=delta_x_3[0])
    f_y = cp.fft.fftfreq(n=shape[1], d=delta_x_3[1])
    f_z = cp.fft.fftfreq(n=shape[2], d=delta_x_3[2])

    kinetic_operator_kernel(
        grid_size,
        block_size,
        (
            P_kinetic,

            cp.float32(delta_x_3[0]),
            cp.float32(delta_x_3[1]),
            cp.float32(delta_x_3[2]),

            cp.float32(delta_time),
            cp.float32(mass),

            f_x,
            f_y,
            f_z
        )
    )
    return P_kinetic


potential_operator_kernel_source = (Path("sources/cuda_kernels/potential_operator.cu")
                                    .read_text().replace("PATH_TO_SOURCES", os.path.abspath("sources")))


def init_potential_operator(P_potential: cp.ndarray, V: cp.ndarray, delta_time:float):
    potential_operator_kernel = cp.RawKernel(potential_operator_kernel_source,
                                 'potential_operator_kernel',
                                 enable_cooperative_groups=False)
    shape = P_potential.shape
    grid_size = math_utils.get_grid_size(shape)
    block_size = (shape[0] // grid_size[0], shape[1] // grid_size[1], shape[2] // grid_size[2])
    potential_operator_kernel(
        grid_size,
        block_size,
        (
            P_potential,
            V,
            cp.float32(delta_time)
        )
    )
    return P_potential