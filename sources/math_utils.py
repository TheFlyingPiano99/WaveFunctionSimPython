import math

planck_constant = 2.0 * math.pi
electron_rest_mass = 1.0
reduced_planck_constant = 1.0
speed_of_light = 137.03599
from numba import jit
import numpy as np


@jit(nopython=True, fastmath=True)
def exp_i(angle):
    return math.cos(angle) + 1j * math.sin(angle)


@jit(nopython=True, fastmath=True, parallel=True)
def square_of_abs(wave_tensor):
    return np.square(np.abs(wave_tensor))


@jit(nopython=True, fastmath=True, parallel=True)
def transform_center_origin_to_corner_origin_system(pos: np.array, box_width: float):
    return pos + np.array([box_width, box_width, box_width]) * 0.5


@jit(nopython=True, fastmath=True, parallel=True)
def transform_corner_origin_to_center_origin_system(pos: np.array, box_width: float):
    return pos - np.array([box_width, box_width, box_width]) * 0.5


def electron_volt_to_hartree(value: float):
    return value * 3.67493 * 10.0 ** (-2)


def angstrom_to_bohr_radii(value):
    return value * 0.52917721067


def kinectic_energy(mass: float, velocity):
    return 0.5 * mass * velocity**2


def get_de_broglie_wave_length_bohr_radii(momentum_h_bar_per_bohr_radius):
    return planck_constant / momentum_h_bar_per_bohr_radius


def classical_momentum(mass: float, velocity):
    return mass * velocity


def relativistic_momentum(rest_mass: float, velocity):
    return rest_mass * velocity / (1.0 - velocity**2 / speed_of_light**2) ** 0.5


def relativistic_energy(momentum, rest_mass: float):
    return ((momentum * speed_of_light) ** 2 + (rest_mass * speed_of_light) ** 2) ** 0.5


def h_bar_per_hartree_to_ns(t: float):
    return t * 2.4188843265857 * 10 ** (-8)
