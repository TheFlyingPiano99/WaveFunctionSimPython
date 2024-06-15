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


class WavePacket:
    _initial_wp_position_bohr_radii_3: np.array = np.array([0.0, 0.0, 0.0])
    _initial_wp_velocity_bohr_radii_hartree_per_h_bar_3: np.array = np.array([0.0, 0.0, 0.0])
    _initial_wp_momentum_h_per_bohr_radii_3: np.array = np.array([0.0, 0.0, 0.0])
    _particle_mass_electron_rest_mass: float = 1.0
    _de_broglie_wave_length_bohr_radii: float

    def get_initial_wp_position_bohr_radii_3(self):
        return self._initial_wp_position_bohr_radii_3

    def get_initial_wp_velocity_bohr_radii_hartree_per_h_bar_3(self):
        return self._initial_wp_velocity_bohr_radii_hartree_per_h_bar_3

    def get_initial_wp_momentum_h_per_bohr_radii_3(self):
        return self._initial_wp_momentum_h_per_bohr_radii_3

    def get_particle_mass_electron_rest_mass(self):
        return self._particle_mass_electron_rest_mass

    def get_de_broglie_wave_length_bohr_radii(self):
        return self._de_broglie_wave_length_bohr_radii

class GaussianWavePacket(WavePacket):
    __initial_wp_width_bohr_radii: float = 2.0

    def __init__(self, config: Dict):
        super().__init__()
        self._particle_mass_electron_rest_mass = try_read_param(config, "wave_packet.particle_mass_electron_rest_mass",
                                                                1.0)
        self._initial_wp_position_bohr_radii_3 = np.array(
            try_read_param(config, "wave_packet.initial_wp_position_bohr_radii_3", [0.0, 0.0, 0.0])
        )
        self._initial_wp_velocity_bohr_radii_hartree_per_h_bar_3 = np.array(
            try_read_param(config, "wave_packet.initial_wp_velocity_bohr_radii_hartree_per_h_bar_3", [0.0, 0.0, 0.0])
        )
        self._initial_wp_momentum_h_per_bohr_radii_3 = math_utils.classical_momentum(
            mass=self._particle_mass_electron_rest_mass,
            velocity=self._initial_wp_velocity_bohr_radii_hartree_per_h_bar_3,
        )
        momentum_magnitude = (
                np.dot(
                    self._initial_wp_momentum_h_per_bohr_radii_3,
                    self._initial_wp_momentum_h_per_bohr_radii_3,
                )
                ** 0.5
        )
        self._de_broglie_wave_length_bohr_radii = (
            math_utils.get_de_broglie_wave_length_bohr_radii(momentum_magnitude)
        )
        self.__initial_wp_width_bohr_radii = try_read_param(config, "wave_packet.initial_wp_width_bohr_radii", 2.0)

    wave_packet_kernel_source = Path("sources/cuda_kernels/gaussian_wave_packet.cu").read_text().replace(
        "PATH_TO_SOURCES", os.path.abspath("sources"))

    def get_initial_wp_width_bohr_radii(self):
        return self.__initial_wp_width_bohr_radii

    def init_wave_packet(
            self,
            delta_x_bohr_radii_3: np.array,
            shape: np.shape,
    ):
        wave_packet_kernel = cp.RawKernel(self.wave_packet_kernel_source,
                                          'wave_packet_kernel',
                                          enable_cooperative_groups=False)
        grid_size = math_utils.get_grid_size(shape)

        block_size = (shape[0] // grid_size[0], shape[1] // grid_size[1], shape[2] // grid_size[2])
        wave_tensor = cp.zeros(shape=shape, dtype=cp.csingle)   # Prepare an empty tensor
        wave_packet_kernel(
            grid_size,
            block_size,
            (
                wave_tensor,

                cp.float32(delta_x_bohr_radii_3[0]),
                cp.float32(delta_x_bohr_radii_3[1]),
                cp.float32(delta_x_bohr_radii_3[2]),

                cp.float32(self.__initial_wp_width_bohr_radii * 2.0),

                cp.float32(self._initial_wp_position_bohr_radii_3[0]),
                cp.float32(self._initial_wp_position_bohr_radii_3[1]),
                cp.float32(self._initial_wp_position_bohr_radii_3[2]),

                cp.float32(self._initial_wp_momentum_h_per_bohr_radii_3[0]),
                cp.float32(self._initial_wp_momentum_h_per_bohr_radii_3[1]),
                cp.float32(self._initial_wp_momentum_h_per_bohr_radii_3[2])
            )
        )
        return wave_tensor
