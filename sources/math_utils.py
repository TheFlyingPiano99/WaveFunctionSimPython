import math

planck_constant = 2.0 * math.pi
reduced_planck_constant = 1.0
electron_rest_mass = 1.0
speed_of_light = 137.03599
from numba import jit
import numpy as np
import cupy as cp
from scipy.spatial.transform import Rotation as R


def exp_i(angle: float):
    return math.cos(angle) + 1j * math.sin(angle)


def exp_i(cangle: np.complex_):
    return (math.cos(cangle.real) + 1j * math.sin(cangle.real)) * math.exp(-cangle.imag)


def square_of_abs(wave_tensor: cp.ndarray):
    return cp.square(cp.abs(wave_tensor))


def vector_length(vec: np.array):
    return np.sqrt(np.dot(vec, vec))


def remap(t: float, min: float, max: float):
    return (t - min) / (max - min)


def clamp(t: float, val0: float, val1: float):
    return min(max(val0, t), val1)


def interpolate(val0: float, val1: float, t: float, exponent: float = 1.0):
    return (1.0 - t ** exponent) * val0 + t ** exponent * val1


def transform_center_origin_to_corner_origin_system(pos: np.array, box_dimensions: np.array):
    return pos + box_dimensions * 0.5


def transform_corner_origin_to_center_origin_system(pos: np.array, box_dimensions: np.array):
    return pos - box_dimensions * 0.5


def electron_volt_to_hartree(value: float):
    return value * 3.67493 * 10.0 ** (-2)


def angstrom_to_bohr_radii(value):
    return value * 0.52917721067


def kinectic_energy(mass: float, velocity):
    return 0.5 * mass * velocity ** 2


def get_de_broglie_wave_length_bohr_radii(momentum_h_bar_per_bohr_radius: float):
    if momentum_h_bar_per_bohr_radius == 0.0:
        return math.inf
    return planck_constant / momentum_h_bar_per_bohr_radius


def classical_momentum(mass: float, velocity: np.array):
    return mass * velocity


def relativistic_momentum(rest_mass: float, velocity):
    return rest_mass * velocity / (1.0 - velocity ** 2 / speed_of_light ** 2) ** 0.5


def relativistic_energy(momentum, rest_mass: float):
    return ((momentum * speed_of_light) ** 2 + (rest_mass * speed_of_light) ** 2) ** 0.5


def h_bar_per_hartree_to_ns(t: float):
    return t * 2.4188843265857 * 10 ** (-8)


def h_bar_per_hartree_to_fs(t: float):
    return h_bar_per_hartree_to_ns(t) * 10 ** 6


def cut_bounding_box(arr: np.ndarray, bottom: np.array, top: np.array):
    return arr[bottom[0]: top[0], bottom[1]: top[1], bottom[2]: top[2]]


def get_momentum_for_harmonic_oscillator(V_max: float, x_max: float):
    return math.sqrt(2.0 * V_max / (4 * electron_rest_mass * speed_of_light ** 2 * x_max ** 2))


def normalize(v: np.array):
    return v / vector_length(v)


def prefered_up():
    return np.array([0.0, 1.0, 0.0], dtype=np.float_)  # +Y


def rotate(vec: np.array, axis: np.array, radians: float):
    r = R.from_rotvec(radians * normalize(axis))
    return r.apply(vec)


# Code source: https://stackoverflow.com/questions/6802577/rotation-of-3d-vector
def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


"""
Source: https://www.quora.com/How-do-you-check-if-a-number-is-a-power-of-2-in-Python
"""


def is_power_of_two(n: int):
    return bin(n).count('1') == 1


def nearest_power_of_2(n: int):
    """
    Source: https://www.geeksforgeeks.org/smallest-power-of-2-greater-than-or-equal-to-n/
    :param n:
    :return: Nearest power of two that is not less than n
    """
    # Calculate log2 of N
    a = int(math.log2(n))
    # If 2^a is equal to N, return N
    if 2 ** a == n:
        return n
    # Return 2^(a + 1)
    return 2 ** (a + 1)


grid_sizes = {}


def get_grid_size_block_size(shape: np.shape, reduced_thread_count: bool = False):
    global grid_sizes
    key = ()
    # Create cache key and use cache if available:
    if len(shape) == 3:
        key = (shape[0], shape[1], shape[2])
        if key in grid_sizes:  # Use cached value
            return tuple(grid_sizes[key]), (shape[0] // grid_sizes[key][0], shape[1] // grid_sizes[key][1], shape[2] // grid_sizes[key][2])
    elif len(shape) == 2:
        key = (shape[0], shape[1])
        if key in grid_sizes:  # Use cached value
            return tuple(grid_sizes[key]), (shape[0] // grid_sizes[key][0], shape[1] // grid_sizes[key][1])
    elif len(shape) == 1:
        key = (shape[0])
        if key in grid_sizes:  # Use cached value
            return tuple(grid_sizes[key]), (shape[0] // grid_sizes[key][0])

    allowed_thread_count_per_block = 256 if reduced_thread_count else 1024
    initial_guess = 2
    while True:
        grid_size = []
        block_size = []
        thread_count = 1
        for i in range(len(shape)):
            grid_size.append(initial_guess)
            if shape[i] < initial_guess:
                grid_size[i] = shape[i]
            # Find divisor:
            while True:
                if shape[i] % grid_size[i] == 0:
                    break
                grid_size[i] += 1
            # Check total thread count in a single block:
            block_dim = shape[i] // grid_size[i]
            thread_count *= block_dim
            block_size.append(block_dim)
        if thread_count <= allowed_thread_count_per_block:
            break
        else:
            initial_guess *= 2  # Need to find a greater divisor

    grid_sizes[key] = grid_size  # Cache

    print(f"For the shape {shape} grid size of {grid_size} and block size of {block_size} was calculated.")
    return tuple(grid_size), tuple(block_size)


def indefinite_simpson_integral(array: np.array, dt: float):
    n = array.size

    # Early termination:
    if n == 0:
        return np.empty(shape=0, dtype=array.dtype)
    if n == 1:
        return np.array([0.0], dtype=array.dtype)
    if n == 2:
        return np.array([0.0, (array[0] + array[1]) / 2.0], dtype=array.dtype)

    # Determine odd and even parts:
    is_even = n % 2 == 0
    odd_section = n
    if is_even:
        odd_section -= 1
    even_section = n
    if not is_even:
        even_section -= 1

    # Create array to accommodate the results:
    integral = np.array(np.zeros(shape=array.size, dtype=array.dtype).tolist())

    # First two elements are calculated using the trapezoidal rule:
    integral[0] = 0.0
    integral[1] = (array[0] + array[1]) / 2.0
    integral[1] = 3.0 * integral[1]  # Compensate for final division

    # Simpson coefficients: [1, 4, 1]
    for i in range(2, odd_section, 2):  # Iterate on odd elements
        integral[i] = integral[i - 2] + array[i - 2] + 4.0 * array[i - 1] + array[i]

    for i in range(3, even_section, 2):  # Iterate on even elements
        integral[i] = integral[i - 2] + array[i - 2] + 4.0 * array[i - 1] + array[i]

    return integral * dt / 3.0


def predict_free_space_standard_devation(delta_t:float, sigma0: float, mass: float, step_count: int)-> np.array:
    """
    Source: https://dx.doi.org/10.1088/0143-0807/18/3/022
    :param delta_t: time resolution
    :param sigma0: initial standard deviation
    :param mass: mass of the particle
    :param step_count: number of discrete steps to calculate the prediction
    :return: a NumPy array of the predicted standard deviations for step_count iterations
    """
    a = sigma0 * 2.0
    array = np.array(np.zeros(step_count, dtype=np.float64).tolist())
    for i in range(step_count):
        t = i * delta_t
        array[i] = math.sqrt(a**2 + 4.0 * (1.0 / mass**2) * t**2 / a**2)
    return array / 2.0


def predict_free_space_expected_location(delta_t:float, x0: float, velocity: float, step_count: int)-> np.array:
    """
    Source: https://dx.doi.org/10.1088/0143-0807/18/3/022
    :param delta_t: time resolution
    :param x0: Initial position
    :param velocity: wave propagation velocity
    :param step_count: number of discrete steps to calculate the prediction
    :return: a NumPy array of the predicted standard deviations for step_count iterations
    """
    array = np.array(np.zeros(step_count, dtype=np.float64).tolist())
    for i in range(step_count):
        t = i * delta_t
        array[i] = x0 + t * velocity
    return array