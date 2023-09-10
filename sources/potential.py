import numpy as np
import math
import sources.math_utils

def init_potential_box(N, delta_x, wall_thickness_bohr_radii, potential_wall_height_hartree):
    V = np.zeros(shape=(N, N, N), dtype=float)
    for x in range(0, N):
        for y in range(0, N):
            for z in range(0, N):
                r = np.array([x, y, z]) * delta_x
                V[x, y, z] = 0.0  # Zero in the middle
                # Barriers:
                # X-axis:
                if r[0] < wall_thickness_bohr_radii:
                    V[x, y, z] = potential_wall_height_hartree * (wall_thickness_bohr_radii - r[0]) / wall_thickness_bohr_radii
                if N * delta_x - r[0] < wall_thickness_bohr_radii:
                    V[x, y, z] = potential_wall_height_hartree * (wall_thickness_bohr_radii - (delta_x * N - r[0])) / wall_thickness_bohr_radii
                # Y-axis:
                if r[1] < wall_thickness_bohr_radii:
                    V[x, y, z] = potential_wall_height_hartree * (wall_thickness_bohr_radii - r[1]) / wall_thickness_bohr_radii
                if N * delta_x - r[1] < wall_thickness_bohr_radii:
                    V[x, y, z] = potential_wall_height_hartree * (wall_thickness_bohr_radii - (delta_x * N - r[1])) / wall_thickness_bohr_radii
                # Z-axis:
                if r[2] < wall_thickness_bohr_radii:
                    V[x, y, z] = potential_wall_height_hartree * (wall_thickness_bohr_radii - (r[2])) / wall_thickness_bohr_radii
                if N * delta_x - r[2] < wall_thickness_bohr_radii:
                    V[x, y, z] = potential_wall_height_hartree * (wall_thickness_bohr_radii - (delta_x * N - r[2])) / wall_thickness_bohr_radii
    return V

def init_potential_sphere(N, delta_x, wall_thickness, potential_wall_hight):
    V = np.zeros(shape=(N, N, N), dtype=float)
    for x in range(0, N):
        for y in range(0, N):
            for z in range(0, N):
                r = np.array([x, y, z]) / np.array([N - 1, N - 1, N - 1])
                V[x, y, z] = 0.0  # Zero in the middle
                # Barriers:
                dir = r - np.array([0.5, 0.5, 0.5])
                l = np.dot(dir, dir) ** 0.5
                if l > (0.5 - wall_thickness):
                    V[x, y, z] = potential_wall_hight * (l - (0.5 - wall_thickness)) / wall_thickness
    return V

def init_zero_potential(N):
    V = np.zeros(shape=(N, N, N), dtype=float)
    return V

def add_wall(V, delta_x, center_bohr_radii, thickness_bohr_radii, height_hartree):
    for x in range(0, V.shape[0]):
        for y in range(0, V.shape[1]):
            for z in range(0, V.shape[2]):
                r = np.array([x, y, z]) * delta_x
                if r[2] > center_bohr_radii - thickness_bohr_radii / 2.0 and r[2] < center_bohr_radii + thickness_bohr_radii / 2.0:
                    v = height_hartree * (1.0 - abs(center_bohr_radii - r[2]) / thickness_bohr_radii * 2.0)
                    V[x, y, z] += v

    return V


def add_single_slit(V, delta_x, center_bohr_radii, thickness_bohr_radii, height_hartree, slit_size_bohr_radii):
    for x in range(0, V.shape[0]):
        for y in range(0, V.shape[1]):
            for z in range(0, V.shape[2]):
                r = np.array([x, y, z]) * delta_x
                if r[2] > center_bohr_radii - thickness_bohr_radii / 2.0 and r[2] < center_bohr_radii + thickness_bohr_radii / 2.0:
                    v = height_hartree * max(0.0, 1.0 - abs(center_bohr_radii - r[2]) / thickness_bohr_radii * 2.0 - max(0.0, 1.0 + 0.02 - (((center_bohr_radii - r[1])**2 + (center_bohr_radii - r[0])**2)**0.5 / slit_size_bohr_radii * 2.0)**2.0))
                    V[x, y, z] += v

    return V