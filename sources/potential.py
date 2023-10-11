import numpy as np
import cupy as cp
import sources.math_utils as math_utils

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
        simulated_volume_width = config["Volume"]["simulated_volume_width_bohr_radii"]
        self.outer_radius_bohr_radii = simulated_volume_width * 0.5
        self.max_potential_hartree = config["Potential"][
            "outer_drain_potential_hartree"
        ]
        self.exponent = config["Potential"]["drain_interpolation_exponent"]


def add_potential_box(
    N, delta_x, wall_thickness_bohr_radii, potential_wall_height_hartree, V : cp.ndarray=None
):
    if V is None:
        V = cp.zeros(shape=(N, N, N), dtype=cp.complex_)
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


def add_draining_potential(
    N : int,
    delta_x : float,
    inner_radius_bohr_radii : float,
    outer_radius_bohr_radii : float,
    max_potential_hartree : float,
    exponent : float,
    V : cp.ndarray,
):
    for x in range(N):
        for y in range(N):
            for z in range(N):
                pos = (
                    cp.array([x, y, z]) * delta_x
                    - cp.array([1.0, 1.0, 1.0]) * N * delta_x * 0.5
                )
                t = min(
                    max(
                        0.0,
                        (cp.sqrt(cp.dot(pos, pos)) - inner_radius_bohr_radii)
                        / (outer_radius_bohr_radii - inner_radius_bohr_radii),
                    ),
                    1.0,
                )
                V[x, y, z] += 1j * t**exponent * max_potential_hartree
    return V


def init_potential_sphere(N : int, delta_x : float, wall_thickness : float, potential_wall_hight : float):
    V = cp.zeros(shape=(N, N, N), dtype=cp.complex_)
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
    V = cp.zeros(shape=(N, N, N), dtype=cp.complex_)
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


def add_double_slit(
    delta_x : float,
    center_bohr_radii : np.array,
    thickness_bohr_radii : float,
    height_hartree : float,
    space_between_slits_bohr_radii : float,
    slit_width_bohr_radii : float,
    shape: cp.shape,
    V: cp.ndarray,
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
