import numba
import numpy as np
import math
import sources.math_utils as math_utils
from numba import jit


class DrainPotentialDescription:
    boundary_bottom_corner_bohr_radii: np.array
    boundary_top_corner_bohr_radii: np.array
    inner_radius_bohr_radii: float  # The greatest distance of a viewing window corner from the origin
    outer_radius_bohr_radii: float  # Maximal radius in simulated cube
    max_potential_hartree: float
    exponent: float

    def __init__(self, config):
        self.boundary_bottom_corner_bohr_radii = np.array(
            config["Volume"]["viewing_window_boundary_bottom_corner_bohr_radii_3"]
        )
        self.boundary_top_corner_bohr_radii = np.array(
            config["Volume"]["viewing_window_boundary_top_corner_bohr_radii_3"]
        )
        self.inner_radius_bohr_radii = max(
            math_utils.vector_length(self.boundary_bottom_corner_bohr_radii),
            math_utils.vector_length(self.boundary_top_corner_bohr_radii),
        )
        simulated_volume_width = config["Volume"]["simulated_volume_width_bohr_radii"]
        self.outer_radius_bohr_radii = simulated_volume_width * 0.5 * 3.0**0.5
        self.max_potential_hartree = config["Potential"][
            "outer_drain_potential_hartree"
        ]
        self.exponent = config["Potential"]["drain_interpolation_exponent"]


@jit(nopython=True)
def add_potential_box(
    N, delta_x, wall_thickness_bohr_radii, potential_wall_height_hartree, V=None
):
    if V is None:
        V = np.zeros(shape=(N, N, N), dtype=np.complex_)
    for x in range(0, V.shape[0]):
        for y in range(0, V.shape[1]):
            for z in range(0, V.shape[2]):
                r = np.array([x, y, z]) * delta_x
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


@jit(nopython=True, fastmath=True)
def add_draining_potential(
    N,
    delta_x,
    inner_radius_bohr_radii,
    outer_radius_bohr_radii,
    max_potential_hartree,
    exponent,
    V,
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


@jit(nopython=True)
def init_potential_sphere(N, delta_x, wall_thickness, potential_wall_hight):
    V = np.zeros(shape=(N, N, N), dtype=np.complex_)
    for x in range(0, N):
        for y in range(0, N):
            for z in range(0, N):
                r = np.array([x, y, z]) / np.array([N - 1, N - 1, N - 1])
                V[x, y, z] = 0.0  # Zero in the middle
                # Barriers:
                dir = r - np.array([0.5, 0.5, 0.5])
                l = np.dot(dir, dir) ** 0.5
                if l > (0.5 - wall_thickness):
                    V[x, y, z] = (
                        potential_wall_hight
                        * (l - (0.5 - wall_thickness))
                        / wall_thickness
                    )
    return V


@jit(nopython=True)
def init_zero_potential(N):
    V = np.zeros(shape=(N, N, N), dtype=np.complex_)
    return V


@jit(nopython=True)
def add_wall(V, delta_x, center_bohr_radii, thickness_bohr_radii, height_hartree):
    for x in range(0, V.shape[0]):
        for y in range(0, V.shape[1]):
            for z in range(0, V.shape[2]):
                r = np.array([x, y, z]) * delta_x
                if (
                    r[2] > center_bohr_radii - thickness_bohr_radii / 2.0
                    and r[2] < center_bohr_radii + thickness_bohr_radii / 2.0
                ):
                    v = height_hartree * (
                        1.0 - abs(center_bohr_radii - r[2]) / thickness_bohr_radii * 2.0
                    )
                    V[x, y, z] += v

    return V


@jit(nopython=True)
def add_single_slit(
    delta_x,
    center_bohr_radii,
    thickness_bohr_radii,
    height_hartree,
    slit_size_bohr_radii,
    V: np.ndarray,
):
    for x in range(0, V.shape[0]):
        for y in range(0, V.shape[1]):
            for z in range(0, V.shape[2]):
                r = np.array([x, y, z]) * delta_x
                if (
                    r[2] > center_bohr_radii - thickness_bohr_radii / 2.0
                    and r[2] < center_bohr_radii + thickness_bohr_radii / 2.0
                ):
                    v = height_hartree * max(
                        0.0,
                        1.0
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


@jit(nopython=True, fastmath=True)
def add_double_slit(
    delta_x,
    center_bohr_radii,
    thickness_bohr_radii,
    height_hartree,
    space_between_slits_bohr_radii,
    slit_width_bohr_radii,
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
