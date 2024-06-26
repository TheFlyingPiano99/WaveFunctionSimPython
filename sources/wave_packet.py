import numpy as np
import cupy as cp
import math
import sources.math_utils as math_utils
from numba import jit, njit
from pathlib import Path
import os
from typing import Dict
from sources.config_read_helper import try_read_param


def P_free_space(r, t):
    return (
            1.0
            / (2.0 * math.pi * t) ** 0.5
            * math.exp(-1j * math.pi / 4)
            * math.exp(1j * cp.dot(r, r) / 2.0 / t)
    )


def wave_0_x(x):
    sum = cp.complex128(0.0)
    for i in range(10):
        sum += P_free_space(cp.array([x, 0]), i)
    return sum


def wave_0_y(y):
    sum = cp.complex128(0.0)
    for i in range(10):
        sum += P_free_space(cp.array([0, y]), i)
    return sum


def wave_packet(x, y):
    return wave_0_x(x) * wave_0_y(y)


class WavePacket:
    _initial_position_bohr_radii_3: np.array = np.array([0.0, 0.0, 0.0])
    _initial_wp_velocity_bohr_radii_hartree_per_h_bar_3: np.array = np.array([0.0, 0.0, 0.0])
    _initial_momentum_h_per_bohr_radii_3: np.array = np.array([0.0, 0.0, 0.0])
    _particle_mass_electron_rest_mass: float = 1.0
    _de_broglie_wave_length_bohr_radii: float


    def get_initial_wp_position_bohr_radii_3(self):
        return self._initial_position_bohr_radii_3

    def get_initial_wp_velocity_bohr_radii_hartree_per_h_bar_3(self):
        return self._initial_wp_velocity_bohr_radii_hartree_per_h_bar_3

    def get_initial_wp_momentum_h_per_bohr_radii_3(self):
        return self._initial_momentum_h_per_bohr_radii_3

    def get_particle_mass_electron_rest_mass(self):
        return self._particle_mass_electron_rest_mass

    def get_de_broglie_wave_length_bohr_radii(self):
        return self._de_broglie_wave_length_bohr_radii

class GaussianWavePacket(WavePacket):
    __initial_standard_deviation_bohr_radii_3: np.array

    def __init__(self, config: Dict):
        super().__init__()
        self._particle_mass_electron_rest_mass = try_read_param(config, "wave_packet.particle_mass_electron_rest_mass",
                                                                1.0)
        self._initial_position_bohr_radii_3 = np.array(
            try_read_param(config, "wave_packet.initial_position_bohr_radii_3", [0.0, 0.0, 0.0])
        )
        self._initial_wp_velocity_bohr_radii_hartree_per_h_bar_3 = np.array(
            try_read_param(config, "wave_packet.initial_velocity_bohr_radii_hartree_per_h_bar_3", [0.0, 0.0, 0.0])
        )
        self._initial_momentum_h_per_bohr_radii_3 = math_utils.classical_momentum(
            mass=self._particle_mass_electron_rest_mass,
            velocity=self._initial_wp_velocity_bohr_radii_hartree_per_h_bar_3,
        )
        momentum_magnitude = (
                np.dot(
                    self._initial_momentum_h_per_bohr_radii_3,
                    self._initial_momentum_h_per_bohr_radii_3,
                )
                ** 0.5
        )
        self._de_broglie_wave_length_bohr_radii = (
            math_utils.get_de_broglie_wave_length_bohr_radii(momentum_magnitude)
        )
        self.__initial_standard_deviation_bohr_radii_3 = np.array(
            try_read_param(config, "wave_packet.initial_standard_deviation_bohr_radii_3", [5.0, 5.0, 5.0])
        )

    def get_initial_standard_deviation_bohr_radii(self):
        return self.__initial_standard_deviation_bohr_radii_3

    def init_wave_packet(
            self,
            delta_x_bohr_radii_3: np.array,
            shape: np.shape,
            double_precision: bool = False
    ):
        wave_packet_kernel_source = (Path("sources/cuda_kernels/gaussian_wave_packet.cu").read_text().replace(
            "PATH_TO_SOURCES", os.path.abspath("sources"))
                                     .replace("T_WF_FLOAT",
                          "double" if double_precision else "float"))

        wave_packet_kernel = cp.RawKernel(wave_packet_kernel_source,
                                          'wave_packet_kernel',
                                          enable_cooperative_groups=False)
        grid_size, block_size = math_utils.get_grid_size_block_size(shape)

        # Prepare an empty tensor:
        wave_tensor = cp.zeros(shape=shape, dtype=(cp.complex128 if double_precision else cp.complex64))
        wave_packet_kernel(
            grid_size,
            block_size,
            (
                wave_tensor,

                (cp.float64(delta_x_bohr_radii_3[0]) if double_precision else cp.float32(delta_x_bohr_radii_3[0])),
                (cp.float64(delta_x_bohr_radii_3[1]) if double_precision else cp.float32(delta_x_bohr_radii_3[1])),
                (cp.float64(delta_x_bohr_radii_3[2]) if double_precision else cp.float32(delta_x_bohr_radii_3[2])),

                (cp.float64(self.__initial_standard_deviation_bohr_radii_3[0]) if double_precision
                    else cp.float32(self.__initial_standard_deviation_bohr_radii_3[0])),
                (cp.float64(self.__initial_standard_deviation_bohr_radii_3[1]) if double_precision
                    else cp.float32(self.__initial_standard_deviation_bohr_radii_3[1])),
                (cp.float64(self.__initial_standard_deviation_bohr_radii_3[2]) if double_precision
                    else cp.float32(self.__initial_standard_deviation_bohr_radii_3[2])),

                (cp.float64(self._initial_position_bohr_radii_3[0]) if double_precision
                    else cp.float32(self._initial_position_bohr_radii_3[0])),
                (cp.float64(self._initial_position_bohr_radii_3[1]) if double_precision
                    else cp.float32(self._initial_position_bohr_radii_3[1])),
                (cp.float64(self._initial_position_bohr_radii_3[2]) if double_precision
                    else cp.float32(self._initial_position_bohr_radii_3[2])),

                (cp.float64(self._initial_momentum_h_per_bohr_radii_3[0]) if double_precision
                    else cp.float32(self._initial_momentum_h_per_bohr_radii_3[0])),
                (cp.float64(self._initial_momentum_h_per_bohr_radii_3[1]) if double_precision
                    else cp.float32(self._initial_momentum_h_per_bohr_radii_3[1])),
                (cp.float64(self._initial_momentum_h_per_bohr_radii_3[2]) if double_precision
                    else cp.float32(self._initial_momentum_h_per_bohr_radii_3[2]))
            )
        )
        return wave_tensor
