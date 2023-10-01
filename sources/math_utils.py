import math

planck_constant = 2.0 * math.pi
electron_rest_mass = 1.0
reduced_planck_constant = 1.0
speed_of_light = 137.03599
from numba import jit


@jit(nopython=True)
def exp_i(angle):
    return math.cos(angle) + 1j * math.sin(angle)

def electron_volt_to_hartree(value):
    return value * 3.67493 * 10.0**(-2)

def angstrom_to_bohr_radii(value):
    return value * 0.52917721067


def kinectic_energy(mass, velocity):
    return 0.5 * mass * velocity**2

def get_de_broglie_wave_length_bohr_radii(momentum_h_bar_per_bohr_radius):
    return planck_constant / momentum_h_bar_per_bohr_radius

def classical_momentum(mass, velocity):
    return mass * velocity

def relativistic_momentum(rest_mass, velocity):
    return rest_mass * velocity / (1.0 - velocity**2 / speed_of_light**2)**0.5

def relativistic_energy(momentum, rest_mass):
    return ((momentum * speed_of_light)**2 + (rest_mass * speed_of_light)**2)**0.5

def h_bar_per_hartree_to_ns(t):
    return t * 2.4188843265857 * 10**(-8)