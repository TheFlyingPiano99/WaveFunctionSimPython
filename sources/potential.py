import numpy as np
import cupy as cp
import sources.math_utils as math_utils
from numba import jit
import math

class DrainPotentialDescription:
    boundary_bottom_corner_bohr_radii: np.array
    boundary_top_corner_bohr_radii: np.array
    inner_radius_bohr_radii: float  # The greatest distance of a viewing window corner from the origin
    outer_radius_bohr_radii: float  # Maximal radius in simulated cube
    max_potential_hartree: float
    exponent: float

    def __init__(self, config):
        self.boundary_bottom_corner_bohr_radii = np.array(
            config["volume"]["viewing_window_boundary_bottom_corner_bohr_radii_3"]
        )
        self.boundary_top_corner_bohr_radii = np.array(
            config["volume"]["viewing_window_boundary_top_corner_bohr_radii_3"]
        )
        # Flip if inverted:
        for i in range(3):
            if (
                self.boundary_bottom_corner_bohr_radii[i]
                > self.boundary_top_corner_bohr_radii[i]
            ):
                temp = self.boundary_bottom_corner_bohr_radii[i]
                self.boundary_bottom_corner_bohr_radii[
                    i
                ] = self.boundary_top_corner_bohr_radii[i]
                self.boundary_top_corner_bohr_radii[i] = temp

        self.inner_radius_bohr_radii = max(
            math_utils.vector_length(self.boundary_bottom_corner_bohr_radii),
            math_utils.vector_length(self.boundary_top_corner_bohr_radii),
        )
        simulated_volume_width = config["volume"]["simulated_volume_width_bohr_radii"]
        self.outer_radius_bohr_radii = simulated_volume_width * 0.5
        self.max_potential_hartree = config["drain"][
            "outer_potential_hartree"
        ]
        self.exponent = config["drain"]["interpolation_exponent"]


def add_potential_box(
    N, delta_x, wall_thickness_bohr_radii, potential_wall_height_hartree, V : cp.ndarray=None
):
    if V is None:
        V = cp.zeros(shape=(N, N, N), dtype=cp.csingle)
    for x in range(0, V.shape[0]):
        for y in range(0, V.shape[1]):
            for z in range(0, V.shape[2]):
                r = cp.array([x, y, z]) * delta_x
                # Barriers:
                t = max(
                    0.0,
                    wall_thickness_bohr_radii - r[0],
                    wall_thickness_bohr_radii - (delta_x * N - r[0]),
                    wall_thickness_bohr_radii - r[1],
                    wall_thickness_bohr_radii - (delta_x * N - r[1]),
                    wall_thickness_bohr_radii - (r[2]),
                    wall_thickness_bohr_radii - (delta_x * N - r[2]),
                )
                V[x, y, z] += (
                    potential_wall_height_hartree * t / wall_thickness_bohr_radii
                )
    return V


@jit(nopython=True)
def add_draining_potential(
    N : int,
    delta_x : float,
    inner_radius_bohr_radii : float,
    outer_radius_bohr_radii : float,
    max_potential_hartree : float,
    exponent : float,
    V : np.ndarray,
):
    for x in range(N):
        for y in range(N):
            for z in range(N):
                pos = (
                    np.array([x, y, z]) * delta_x
                    - np.array([1.0, 1.0, 1.0]) * N * delta_x * 0.5
                )
                t = min(
                    max(
                        0.0,
                        (np.sqrt(np.dot(pos, pos)) - inner_radius_bohr_radii)
                        / (outer_radius_bohr_radii - inner_radius_bohr_radii),
                    ),
                    1.0,
                )
                V[x, y, z] += 1j * t**exponent * max_potential_hartree
    return V


def init_potential_sphere(N : int, delta_x : float, wall_thickness : float, potential_wall_hight : float):
    V = cp.zeros(shape=(N, N, N), dtype=cp.csingle)
    for x in range(0, N):
        for y in range(0, N):
            for z in range(0, N):
                r = cp.array([x, y, z]) / cp.array([N - 1, N - 1, N - 1])
                V[x, y, z] = 0.0  # Zero in the middle
                # Barriers:
                dir = r - cp.array([0.5, 0.5, 0.5])
                l = cp.dot(dir, dir) ** 0.5
                if l > (0.5 - wall_thickness):
                    V[x, y, z] = (
                        potential_wall_hight
                        * (l - (0.5 - wall_thickness))
                        / wall_thickness
                    )
    return V


def init_zero_potential(N : int):
    V = cp.zeros(shape=(N, N, N), dtype=cp.csingle)
    return V


def add_wall(V : cp.ndarray, delta_x : float, center_bohr_radii : float, thickness_bohr_radii : float, height_hartree : float):
    for x in range(0, V.shape[0]):
        for y in range(0, V.shape[1]):
            for z in range(0, V.shape[2]):
                r = cp.array([x, y, z]) * delta_x
                if (
                    r[2] > center_bohr_radii - thickness_bohr_radii / 2.0
                    and r[2] < center_bohr_radii + thickness_bohr_radii / 2.0
                ):
                    v = height_hartree * (
                        1.0 - abs(center_bohr_radii - r[2]) / thickness_bohr_radii * 2.0
                    )
                    V[x, y, z] += v

    return V


def add_single_slit(
    delta_x : float,
    center_bohr_radii : float,
    thickness_bohr_radii : float,
    height_hartree : float,
    slit_size_bohr_radii : float,
    V: cp.ndarray,
):
    for x in range(0, V.shape[0]):
        for y in range(0, V.shape[1]):
            for z in range(0, V.shape[2]):
                r = cp.array([x, y, z]) * delta_x
                if (
                    r[2] > center_bohr_radii - thickness_bohr_radii / 2.0
                    and r[2] < center_bohr_radii + thickness_bohr_radii / 2.0
                ):
                    v = height_hartree * max(
                        0.0,
                        0.0
                        - abs(center_bohr_radii - r[2]) / thickness_bohr_radii * 2.0
                        - max(
                            0.0,
                            1.0
                            + 0.02
                            - (
                                (
                                    (center_bohr_radii - r[1]) ** 2
                                    + (center_bohr_radii - r[0]) ** 2
                                )
                                ** 0.5
                                / slit_size_bohr_radii
                                * 2.0
                            )
                            ** 2.0,
                        ),
                    )
                    V[x, y, z] += v

    return V


@jit(nopython=True)
def add_double_slit(
    delta_x : float,
    center_bohr_radii : np.array,
    thickness_bohr_radii : float,
    height_hartree : float,
    space_between_slits_bohr_radii : float,
    slit_width_bohr_radii : float,
    shape: np.shape,
    V: np.ndarray,
):
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                r = math_utils.transform_corner_origin_to_center_origin_system(
                    np.array([x, y, z]) * delta_x, delta_x * shape[0]
                )
                if (
                    r[0] > center_bohr_radii[0] - thickness_bohr_radii / 2.0
                    and r[0] < center_bohr_radii[0] + thickness_bohr_radii / 2.0
                    and not (
                        (
                            r[2]
                            > center_bohr_radii[2]
                            - space_between_slits_bohr_radii * 0.5
                            - slit_width_bohr_radii
                            and r[2]
                            < center_bohr_radii[2]
                            - space_between_slits_bohr_radii * 0.5
                        )
                        or (
                            r[2]
                            < center_bohr_radii[2]
                            + space_between_slits_bohr_radii * 0.5
                            + slit_width_bohr_radii
                            and r[2]
                            > center_bohr_radii[2]
                            + space_between_slits_bohr_radii * 0.5
                        )
                    )
                ):
                    V[x, y, z] += height_hartree

    return V


@jit(nopython=True)
def particle_hard_interaction_potential(V: np.ndarray, delta_x: float, particle_radius_bohr_radius: float, potential_hartree: float):
    for x in range(V.shape[0]):
        for y in range(V.shape[1]):
            for z in range(V.shape[2]):
                r = math_utils.transform_corner_origin_to_center_origin_system(
                    np.array([x, y, z]) * delta_x, delta_x * V.shape[0]
                )
                if (
                    abs(r[0] - r[1]) < 2.0 * particle_radius_bohr_radius
                or abs(r[1] - r[2]) < 2.0 * particle_radius_bohr_radius
                or abs(r[2] - r[0]) < 2.0 * particle_radius_bohr_radius
                ):
                    V[x, y, z] += potential_hartree
    return  V

@jit(nopython=True)
def particle_inv_square_interaction_potential(V: np.ndarray, delta_x: float, potential_hartree: float):
    for x in range(V.shape[0]):
        for y in range(V.shape[1]):
            for z in range(V.shape[2]):
                r = math_utils.transform_corner_origin_to_center_origin_system(
                    np.array([x, y, z]) * delta_x, delta_x * V.shape[0]
                )
                V[x, y, z] += (potential_hartree *
                               (1.0 / max(abs(r[0] - r[1]), 0.0000001) + 1.0 / max(abs(r[0] - r[2]), 0.0000001) + 1.0 / max(abs(r[1] - r[2]), 0.0000001)))
    return V

@jit(nopython=True)
def particle_square_interaction_potential(V: np.ndarray, delta_x: float, potential_hartree: float):
    for x in range(V.shape[0]):
        for y in range(V.shape[1]):
            for z in range(V.shape[2]):
                r = math_utils.transform_corner_origin_to_center_origin_system(
                    np.array([x, y, z]) * delta_x, delta_x * V.shape[0]
                )
                V[x, y, z] += (potential_hartree * (abs(r[0] - r[1]) + abs(r[0] - r[2]) + abs(r[1] - r[2])))
    return V


@jit(nopython=True)
def add_harmonic_oscillator_for_1D(V: np.ndarray, delta_x: float, angular_frequency: float):
    const = 0.5 * math_utils.electron_rest_mass * angular_frequency**2
    for x in range(V.shape[0]):
        for y in range(V.shape[1]):
            for z in range(V.shape[2]):
                r = math_utils.transform_corner_origin_to_center_origin_system(
                    np.array([x, y, z]) * delta_x, delta_x * V.shape[0]
                )
                V[x, y, z] += const * (r[0]* r[0] + r[1] * r[1] + r[2] * r[2])
    return  V


@jit(nopython=True)
def add_wall_for_1D(V: np.ndarray, delta_x: float, center_bohr_radius: float, thickness_bohr_radius: float, potential_hartree: float):
    for x in range(V.shape[0]):
        for y in range(V.shape[1]):
            for z in range(V.shape[2]):
                r = math_utils.transform_corner_origin_to_center_origin_system(
                    np.array([x, y, z]) * delta_x, delta_x * V.shape[0]
                )
                if (r[0] >= center_bohr_radius - thickness_bohr_radius * 0.5
                    and r[0] <= center_bohr_radius + thickness_bohr_radius * 0.5):
                        V[x, y, z] += potential_hartree
                if (r[1] >= center_bohr_radius - thickness_bohr_radius * 0.5
                        and r[1] <= center_bohr_radius + thickness_bohr_radius * 0.5):
                    V[x, y, z] += potential_hartree
                if (r[2] >= center_bohr_radius - thickness_bohr_radius * 0.5
                        and r[2] <= center_bohr_radius + thickness_bohr_radius * 0.5):
                    V[x, y, z] += potential_hartree
    return  V

@jit(nopython=True)
def add_wall(V: np.ndarray, delta_x: float, center_bohr_radius: np.array, normal: np.array, thickness_bohr_radius: float, potential_hartree: float):
    for x in range(V.shape[0]):
        for y in range(V.shape[1]):
            for z in range(V.shape[2]):
                r = math_utils.transform_corner_origin_to_center_origin_system(
                    np.array([x, y, z]) * delta_x, delta_x * V.shape[0]
                )
                d = np.dot(normal, r - center_bohr_radius)
                if d <= thickness_bohr_radius * 0.5 and d >= -thickness_bohr_radius * 0.5:
                    V[x, y, z] += potential_hartree
    return  V

@jit(nopython=True)
def add_coulomb_potential(V: np.ndarray, delta_x: float, center_bohr_radius: np.array, normal: np.array, charge: float):
    for x in range(V.shape[0]):
        for y in range(V.shape[1]):
            for z in range(V.shape[2]):
                r = math_utils.transform_corner_origin_to_center_origin_system(
                    np.array([x, y, z]) * delta_x, delta_x * V.shape[0]
                )
                V[x, y, z] += -charge * max(abs(np.dot(normal, r - center_bohr_radius)), 0.00000001)
    return  V

@jit(nopython=True)
def add_optical_grid(V: np.ndarray, delta_x: float, center_bohr_radius: np.array, normal: np.array, distance_between_nodes_bohr_radii: float, potential_hartree: float, node_count: int):
    for x in range(V.shape[0]):
        for y in range(V.shape[1]):
            for z in range(V.shape[2]):
                r = math_utils.transform_corner_origin_to_center_origin_system(
                    np.array([x, y, z]) * delta_x, delta_x * V.shape[0]
                )
                right = np.cross(normal, math_utils.prefered_up())
                up = np.cross(right, normal)
                if (abs(np.dot(normal, r - center_bohr_radius)) < 10.0
                    and abs(np.dot(right, r - center_bohr_radius)) < distance_between_nodes_bohr_radii * node_count * 0.6
                    and abs(np.dot(up, r - center_bohr_radius)) < distance_between_nodes_bohr_radii * node_count * 0.6
                ):
                    for u in range(node_count):
                        for v in range(node_count):
                            current_center = (
                                center_bohr_radius
                                + distance_between_nodes_bohr_radii
                                * (right * float(u - float(node_count) * 0.5) + up * float(v - float(node_count) * 0.5))
                            )

                            d = math_utils.vector_length(current_center - r)
                            V[x, y, z] += potential_hartree * math.exp(-0.5 * d**2)
    return  V

