import numpy as np
import cupy as cp
import sources.math_utils as math_utils
from numba import jit
import math
import sources.potential_kernels as kernels


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


def add_draining_potential(
    V : cp.ndarray,
    delta_x : float,
    inner_radius_bohr_radii : float,
    outer_radius_bohr_radii : float,
    max_potential_hartree : float,
    exponent : float,
):
    draining_potential_kernel = cp.RawKernel(kernels.draining_potential_kernel_source,
                                 'draining_potential_kernel',
                                 enable_cooperative_groups=False)
    shape = V.shape
    grid_size = (64, 64, 64)
    block_size = (shape[0] // grid_size[0], shape[1] // grid_size[1], shape[2] // grid_size[2])
    draining_potential_kernel(
        grid_size,
        block_size,
        (
            V,
            cp.float32(delta_x),
            cp.float32(inner_radius_bohr_radii),
            cp.float32(outer_radius_bohr_radii),
            cp.float32(max_potential_hartree),
            cp.float32(exponent),
        )
    )

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
    V: cp.ndarray,
    delta_x : float,
    center_bohr_radii_3 : np.array,
    thickness_bohr_radii : float,
    potential_hartree : float,
    space_between_slits_bohr_radii : float,
    slit_width_bohr_radii : float
):
    double_slit_kernel = cp.RawKernel(kernels.double_slit_kernel_source,
                                         "double_slit_kernel",
                                         enable_cooperative_groups=False)

    shape = V.shape
    grid_size = (64, 64, 64)
    block_size = (shape[0] // grid_size[0], shape[1] // grid_size[1], shape[2] // grid_size[2])
    double_slit_kernel(
        grid_size,
        block_size,
        (
            V,
            cp.float32(delta_x),

            cp.float32(center_bohr_radii_3[0]),
            cp.float32(center_bohr_radii_3[1]),
            cp.float32(center_bohr_radii_3[2]),

            cp.float32(thickness_bohr_radii),
            cp.float32(potential_hartree),
            cp.float32(space_between_slits_bohr_radii),
            cp.float32(slit_width_bohr_radii),
        )
    )
    return V


def particle_hard_interaction_potential(V: cp.ndarray, delta_x: float, particle_radius_bohr_radius: float, potential_hartree: float):
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

def particle_inv_square_interaction_potential(V: cp.ndarray, delta_x: float, potential_hartree: float):
    for x in range(V.shape[0]):
        for y in range(V.shape[1]):
            for z in range(V.shape[2]):
                r = math_utils.transform_corner_origin_to_center_origin_system(
                    np.array([x, y, z]) * delta_x, delta_x * V.shape[0]
                )
                V[x, y, z] += (potential_hartree *
                               (1.0 / max(abs(r[0] - r[1]), 0.0000001) + 1.0 / max(abs(r[0] - r[2]), 0.0000001) + 1.0 / max(abs(r[1] - r[2]), 0.0000001)))
    return V

def particle_square_interaction_potential(V: cp.ndarray, delta_x: float, potential_hartree: float):
    for x in range(V.shape[0]):
        for y in range(V.shape[1]):
            for z in range(V.shape[2]):
                r = math_utils.transform_corner_origin_to_center_origin_system(
                    np.array([x, y, z]) * delta_x, delta_x * V.shape[0]
                )
                V[x, y, z] += (potential_hartree * (abs(r[0] - r[1]) + abs(r[0] - r[2]) + abs(r[1] - r[2])))
    return V


def add_harmonic_oscillator_for_1D(V: cp.ndarray, delta_x: float, angular_frequency: float):
    const = 0.5 * math_utils.electron_rest_mass * angular_frequency**2
    for x in range(V.shape[0]):
        for y in range(V.shape[1]):
            for z in range(V.shape[2]):
                r = math_utils.transform_corner_origin_to_center_origin_system(
                    np.array([x, y, z]) * delta_x, delta_x * V.shape[0]
                )
                V[x, y, z] += const * (r[0]* r[0] + r[1] * r[1] + r[2] * r[2])
    return  V


def add_wall_for_1D(V: cp.ndarray, delta_x: float, center_bohr_radius: float, thickness_bohr_radius: float, potential_hartree: float):
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

def add_wall(
        V: cp.ndarray,
        delta_x: float,
        center_bohr_radii_3: np.array,
        normal_bohr_radii_3: np.array,
        thickness_bohr_radius: float,
        potential_hartree: float,
     ):
    potential_wall_kernel = cp.RawKernel(kernels.potential_wall_kernel_source,
                                         "potential_wall_kernel",
                                         enable_cooperative_groups=False)

    shape = V.shape
    grid_size = (64, 64, 64)
    block_size = (shape[0] // grid_size[0], shape[1] // grid_size[1], shape[2] // grid_size[2])
    potential_wall_kernel(
        grid_size,
        block_size,
        (
            V,
            cp.float32(delta_x),

            cp.float32(center_bohr_radii_3[0]),
            cp.float32(center_bohr_radii_3[1]),
            cp.float32(center_bohr_radii_3[2]),

            cp.float32(normal_bohr_radii_3[0]),
            cp.float32(normal_bohr_radii_3[1]),
            cp.float32(normal_bohr_radii_3[2]),

            cp.float32(thickness_bohr_radius),
            cp.float32(potential_hartree)
        )
    )
    return V


def add_coulomb_potential(V: cp.ndarray, delta_x: float, center_bohr_radius: np.array, gradient_dir: np.array, charge_density: float,
                          oxide_start_bohr_radii: float, oxide_end_bohr_radii: float):
    for x in range(V.shape[0]):
        for y in range(V.shape[1]):
            for z in range(V.shape[2]):
                r = math_utils.transform_corner_origin_to_center_origin_system(
                    np.array([x, y, z]) * delta_x, delta_x * V.shape[0]
                )
                d = np.dot(gradient_dir, center_bohr_radius - r)
                epsilon = 1.0
                if (d >= oxide_start_bohr_radii and d <= oxide_end_bohr_radii):
                    V[x, y, z] += charge_density / epsilon / 2.0 / max(d, 0.00000001) - charge_density / epsilon / 2.0 / oxide_start_bohr_radii
                elif (d > oxide_end_bohr_radii):
                    V[x, y, z] += charge_density / epsilon / 2.0 / oxide_end_bohr_radii - charge_density / epsilon / 2.0 / oxide_start_bohr_radii
    return  V


def add_linear_potential_gradient(V: cp.ndarray, delta_x: float, center_bohr_radius: np.array, gradient_dir: np.array, gradient_val: float):
    for x in range(V.shape[0]):
        for y in range(V.shape[1]):
            for z in range(V.shape[2]):
                r = math_utils.transform_corner_origin_to_center_origin_system(
                    np.array([x, y, z]) * delta_x, delta_x * V.shape[0]
                )
                V[x, y, z] -= gradient_val * abs(np.dot(gradient_dir, r - center_bohr_radius))
    return  V


def add_optical_grid(V: cp.ndarray, delta_x: float, center_bohr_radius: np.array, normal: np.array, distance_between_nodes_bohr_radii: float, potential_hartree: float, node_count: int):
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


def generate_potential_from_walls_and_drain(V: cp.ndarray, V_vis: cp.ndarray, delta_x, drain_description: DrainPotentialDescription, walls: []):
    V.fill(0.0)
    for w in walls:
        add_wall(
            V=V,
            delta_x=delta_x,
            center_bohr_radii_3=w.center_bohr_radii_3,
            normal_bohr_radii_3=w.normal_bohr_radii_3,
            thickness_bohr_radius=w.thickness_bohr_radii,
            potential_hartree=w.potential_hartree
        )
    V_vis = V.copy()
    add_draining_potential(V=V,
                           delta_x=delta_x,
                           inner_radius_bohr_radii=drain_description.inner_radius_bohr_radii,
                           outer_radius_bohr_radii=drain_description.outer_radius_bohr_radii,
                           max_potential_hartree=drain_description.max_potential_hartree,
                           exponent=drain_description.exponent
                           )

    center_bohr_radii_3: np.array = np.array([0.0, 0.0, 0.0])
    normal_bohr_radii_3: np.array = np.array([1.0, 0.0, 0.0])
    thickness_bohr_radii: float = 5.0
    potential_hartree: float = 20.0
    velocity_bohr_radius_hartree_per_h_bar: np.array = np.array([0.0, 0.0, 0.0])
    angular_velocity_rad_hartree_per_h_bar: np.array = np.array([0.0, 0.0, 0.0])
    potential_change_rate_hartree_2_per_h_bar: float = 0.0

    return V, V_vis