from numba import jit, vectorize
import numpy as np
import cupy as cp
import sources.math_utils as math_utils
from pathlib import Path
import os


kinetic_operator_kernel_source = (Path("sources/cuda_kernels/kinetic_operator.cu")
                                  .read_text().replace("PATH_TO_SOURCES", os.path.abspath("sources")))


def init_kinetic_operator(delta_x_3: np.array, delta_time: float, shape: np.shape):
    kinetic_operator_kernel = cp.RawKernel(kinetic_operator_kernel_source,
                                 'kinetic_operator_kernel',
                                 enable_cooperative_groups=False)
    grid_size = math_utils.get_grid_size(shape)
    block_size = (shape[0] // grid_size[0], shape[1] // grid_size[1], shape[2] // grid_size[2])
    P_kinetic = cp.zeros(shape=shape, dtype=cp.csingle)
    kinetic_operator_kernel(
        grid_size,
        block_size,
        (
            P_kinetic,

            cp.float32(delta_x_3[0]),
            cp.float32(delta_x_3[1]),
            cp.float32(delta_x_3[2]),

            cp.float32(delta_time)
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
