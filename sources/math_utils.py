import math

def exp_i(angle):
    return math.cos(angle) + 1j * math.sin(angle)

def electron_volt_to_hartree(value):
    return value * 3.67493 * 10.0**(-2)

def angstrom_to_bohr_radii(value):
    return value * 0.52917721067