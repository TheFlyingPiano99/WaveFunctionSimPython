import math

planck_constant = 2.0 * math.pi
electron_rest_mass = 1.0
reduced_planck_constant = 1.0
speed_of_light = 137.03599
from numba import jit
import numpy as np
import cupy as cp


@jit(nopython=True, fastmath=True)
def exp_i(angle: float):
    return math.cos(angle) + 1j * math.sin(angle)


@jit(nopython=True, fastmath=True)
def exp_i(cangle: np.complex_):
    return (math.cos(cangle.real) + 1j * math.sin(cangle.real)) * math.exp(-cangle.imag)


def square_of_abs(wave_tensor: cp.ndarray):
    return cp.square(cp.abs(wave_tensor))

def vector_length(vec: np.array):
    return np.sqrt(np.dot(vec, vec))


@jit(nopython=True)
def remap(t: float, min: float, max: float):
    return (t - min) / (max - min)


@jit(nopython=True)
def clamp(t: float, val0: float, val1: float):
    return min(max(val0, t), val1)


@jit(nopython=True)
def interpolate(val0: float, val1: float, t: float, exponent: float = 1.0):
    return (1.0 - t**exponent) * val0 + t**exponent * val1


@jit(nopython=True)
def transform_center_origin_to_corner_origin_system(pos: np.array, box_width: float):
    return pos + np.array([box_width, box_width, box_width]) * 0.5


@jit(nopython=True)
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


def classical_momentum(mass: float, velocity : np.array):
    return mass * velocity


def relativistic_momentum(rest_mass: float, velocity):
    return rest_mass * velocity / (1.0 - velocity**2 / speed_of_light**2) ** 0.5


def relativistic_energy(momentum, rest_mass: float):
    return ((momentum * speed_of_light) ** 2 + (rest_mass * speed_of_light) ** 2) ** 0.5


def h_bar_per_hartree_to_ns(t: float):
    return t * 2.4188843265857 * 10 ** (-8)


def cut_window(arr: np.ndarray, bottom: cp.array, top: np.array):
    return arr[bottom[0] : top[0], bottom[1] : top[1], bottom[2] : top[2]]
