import numpy as np
import cupy as cp
import sources.math_utils as math_utils
from numba import jit
import math
import sources.potential_kernels as kernels
from typing import Dict
from sources.config_read_helper import try_read_param
from colorama import Fore, Style


class PreInitializedPotential:
    path: str
    enable: bool
    visible: bool

    def __init__(self, config: Dict):
        self.path = try_read_param(config, "potential.pre_initialized_potential.path", "")
        self.enable = try_read_param(config, "potential.pre_initialized_potential.enable", False)
        self.visible = try_read_param(config, "potential.pre_initialized_potential.visible", True)

class AbsorbingBoundaryCondition:
    boundary_bottom_corner_bohr_radii_3: np.array
    boundary_top_corner_bohr_radii_3: np.array

    outer_potential_in_positive_xyz_direction_hartree_3: np.array
    outer_potential_in_negative_xyz_direction_hartree_3: np.array

    start_offset: float

    slope_exponent_in_positive_xyz_direction_3: np.array
    slope_exponent_in_negative_xyz_direction_3: np.array

    enable: bool = True

    def __init__(self, config, bottom_corner: np.array, top_corner: np.array):
        self.boundary_bottom_corner_bohr_radii_3 = bottom_corner
        self.boundary_top_corner_bohr_radii_3 = top_corner
        self.outer_potential_in_positive_xyz_direction_hartree_3 = np.array(
            try_read_param(
            config,
            "absorbing_boundary_condition.outer_potential_in_positive_xyz_direction_hartree_3",
            [-100.0, -100.0, -100.0])
        )
        self.outer_potential_in_negative_xyz_direction_hartree_3 = np.array(
            try_read_param(
                config,
            "absorbing_boundary_condition.outer_potential_in_negative_xyz_direction_hartree_3",
            [-100.0, -100.0, -100.0])
        )
        self.start_offset = try_read_param(
            config,
            "absorbing_boundary_condition.start_offset",
            1.0
        )
        self.slope_exponent_in_positive_xyz_direction_3 = np.array(
            try_read_param(
                config,
                "absorbing_boundary_condition.slope_exponent_in_positive_xyz_direction_3",
                [2.0, 2.0, 2.0]
            )
        )
        self.slope_exponent_in_negative_xyz_direction_3 = np.array(
            try_read_param(
                config,
                "absorbing_boundary_condition.slope_exponent_in_negative_xyz_direction_3",
                [2.0, 2.0, 2.0]
            )
        )

        self.enable = try_read_param(config, "absorbing_boundary_condition.enable", True)

    def add_potential(
            self,
            V: cp.ndarray,
            delta_x_3: np.array
    ):
        if not self.enable: # Do not add if disabled
            return V
        absorbing_potential_kernel = cp.RawKernel(kernels.absorbing_potential_kernel_source,
                                                 'absorbing_potential_kernel',
                                                 enable_cooperative_groups=False)
        shape = V.shape
        grid_size = math_utils.get_grid_size(shape)
        block_size = (shape[0] // grid_size[0], shape[1] // grid_size[1], shape[2] // grid_size[2])
        absorbing_potential_kernel(
            grid_size,
            block_size,
            (
                V,

                cp.float32(delta_x_3[0]),
                cp.float32(delta_x_3[1]),
                cp.float32(delta_x_3[2]),

                cp.float32(self.boundary_bottom_corner_bohr_radii_3[0]),
                cp.float32(self.boundary_bottom_corner_bohr_radii_3[1]),
                cp.float32(self.boundary_bottom_corner_bohr_radii_3[2]),

                cp.float32(self.boundary_top_corner_bohr_radii_3[0]),
                cp.float32(self.boundary_top_corner_bohr_radii_3[1]),
                cp.float32(self.boundary_top_corner_bohr_radii_3[2]),

                cp.float32(self.start_offset),

                cp.float32(self.outer_potential_in_positive_xyz_direction_hartree_3[0]),
                cp.float32(self.outer_potential_in_positive_xyz_direction_hartree_3[1]),
                cp.float32(self.outer_potential_in_positive_xyz_direction_hartree_3[2]),

                cp.float32(self.outer_potential_in_negative_xyz_direction_hartree_3[0]),
                cp.float32(self.outer_potential_in_negative_xyz_direction_hartree_3[1]),
                cp.float32(self.outer_potential_in_negative_xyz_direction_hartree_3[2]),

                cp.float32(self.slope_exponent_in_positive_xyz_direction_3[0]),
                cp.float32(self.slope_exponent_in_positive_xyz_direction_3[1]),
                cp.float32(self.slope_exponent_in_positive_xyz_direction_3[2]),

                cp.float32(self.slope_exponent_in_negative_xyz_direction_3[0]),
                cp.float32(self.slope_exponent_in_negative_xyz_direction_3[1]),
                cp.float32(self.slope_exponent_in_negative_xyz_direction_3[2]),
            )
        )

        return V


class PotentialWall:
    potential_hartree: float = 20.0
    center_bohr_radii_3: np.array = np.array([0.0, 0.0, 0.0])
    normal_vector_3: np.array = np.array([1.0, 0.0, 0.0])
    plateau_thickness_bohr_radii: float = 5.0
    slope_thickness_bohr_radii: float = 0.0
    slope_exponent: float = 1.0
    velocity_bohr_radius_hartree_per_h_bar_3: np.array = np.array([0.0, 0.0, 0.0])
    angular_velocity_rad_hartree_per_h_bar_3: np.array = np.array([0.0, 0.0, 0.0])
    potential_change_rate_hartree_sqr_per_h_bar: float = 0.0
    slit_count: int = 0
    slit_spacing_bohr_radii: float = 1.0
    slit_width_bohr_radii: float = 0.5
    slit_rotation_radian: float = 0.0
    visible: bool = True

    def __init__(self, wall_config: Dict):
        self.potential_hartree = try_read_param(wall_config, "potential_hartree", 20)
        self.center_bohr_radii_3 = np.array(try_read_param(wall_config, "center_bohr_radii_3", [0.0, 0.0, 0.0]))
        self.normal_vector_3 = math_utils.normalize(np.array(try_read_param(wall_config, "normal_vector_3", [1.0, 0.0, 0.0])))
        self.plateau_thickness_bohr_radii = try_read_param(wall_config, "plateau_thickness_bohr_radii", 5.0)
        self.slope_thickness_bohr_radii = try_read_param(wall_config, "slope_thickness_bohr_radii", 0.0)
        self.slope_exponent = try_read_param(wall_config, "slope_exponent", 1.0)
        self.velocity_bohr_radius_hartree_per_h_bar_3 = np.array(try_read_param(wall_config, "velocity_bohr_radius_hartree_per_h_bar_3", [0.0, 0.0, 0.0]))
        self.angular_velocity_rad_hartree_per_h_bar_3 = np.array(try_read_param(wall_config, "angular_velocity_rad_hartree_per_h_bar_3", [0.0, 0.0, 0.0]))
        self.potential_change_rate_hartree_sqr_per_h_bar = try_read_param(wall_config, "potential_change_rate_hartree_sqr_per_h_bar", 0.0)
        self.slit_count = try_read_param(wall_config, "slit_count", 0)
        if self.slit_count > 4:
            print(Fore.RED + "Potential wall slit count is currently supported only up to 4 slits!" + Style.RESET_ALL)
            self.slit_count = 4
        elif self.slit_count < 0:
            print(Fore.RED + "Negative potential wall slit count is invalid!" + Style.RESET_ALL)
            self.slit_count = 0
        self.slit_spacing_bohr_radii = try_read_param(wall_config, "slit_spacing_bohr_radii", 1.0)
        self.slit_width_bohr_radii = try_read_param(wall_config, "slit_width_bohr_radii", 0.5)
        self.slit_rotation_radian = try_read_param(wall_config, "slit_rotation_radian", 0.0)
        self.visible = try_read_param(wall_config, "visible", True)
        self.enable = try_read_param(wall_config, "enable", True)


    def add_potential(
            self,
            V: cp.ndarray,
            delta_x_3: np.array,
    ):
        if not self.enable: # Do not add if disabled
            return V
        potential_wall_kernel = cp.RawKernel(kernels.potential_wall_kernel_source,
                                             "potential_wall_kernel",
                                             enable_cooperative_groups=False)

        shape = V.shape
        grid_size = math_utils.get_grid_size(shape)
        block_size = (shape[0] // grid_size[0], shape[1] // grid_size[1], shape[2] // grid_size[2])
        potential_wall_kernel(
            grid_size,
            block_size,
            (
                V,
                cp.float32(self.potential_hartree),

                cp.float32(delta_x_3[0]),
                cp.float32(delta_x_3[1]),
                cp.float32(delta_x_3[2]),

                cp.float32(self.center_bohr_radii_3[0]),
                cp.float32(self.center_bohr_radii_3[1]),
                cp.float32(self.center_bohr_radii_3[2]),

                cp.float32(self.normal_vector_3[0]),
                cp.float32(self.normal_vector_3[1]),
                cp.float32(self.normal_vector_3[2]),

                cp.float32(self.plateau_thickness_bohr_radii),
                cp.float32(self.slope_thickness_bohr_radii),
                cp.float32(self.slope_exponent),
                cp.uint32(self.slit_count),
                cp.float32(self.slit_spacing_bohr_radii),
                cp.float32(self.slit_width_bohr_radii),
                cp.float32(self.slit_rotation_radian)
            )
        )
        return V


def add_potential_box(
        voxels_3, delta_x_3, wall_thickness_bohr_radii, potential_wall_height_hartree, V: cp.ndarray = None
):
    if V is None:
        V = cp.zeros(shape=voxels_3, dtype=cp.complex64)
    for x in range(0, V.shape[0]):
        for y in range(0, V.shape[1]):
            for z in range(0, V.shape[2]):
                r = np.array([x, y, z]) * delta_x_3
                # Barriers:
                t = max(
                    0.0,
                    wall_thickness_bohr_radii - r[0],
                    wall_thickness_bohr_radii - (delta_x_3[0] * voxels_3[0] - r[0]),
                    wall_thickness_bohr_radii - r[1],
                    wall_thickness_bohr_radii - (delta_x_3[1] * voxels_3[1] - r[1]),
                    wall_thickness_bohr_radii - (r[2]),
                    wall_thickness_bohr_radii - (delta_x_3[2] * voxels_3[2] - r[2]),
                )
                V[x, y, z] += (
                        potential_wall_height_hartree * t / wall_thickness_bohr_radii
                )
    return V


def init_potential_sphere(N: int, delta_x_3: np.array, wall_thickness: float, potential_wall_hight: float):
    V = cp.zeros(shape=(N, N, N), dtype=cp.complex64)
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


def init_zero_potential(N: int):
    V = cp.zeros(shape=(N, N, N), dtype=cp.complex64)
    return V


def add_single_slit(
        delta_x_3: np.array,
        center_bohr_radii: float,
        thickness_bohr_radii: float,
        height_hartree: float,
        slit_size_bohr_radii: float,
        V: cp.ndarray,
):
    for x in range(0, V.shape[0]):
        for y in range(0, V.shape[1]):
            for z in range(0, V.shape[2]):
                r = np.array([x, y, z]) * delta_x_3
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
        delta_x_3: np.array,
        center_bohr_radii_3: np.array,
        thickness_bohr_radii: float,
        potential_hartree: float,
        space_between_slits_bohr_radii: float,
        slit_width_bohr_radii: float
):
    double_slit_kernel = cp.RawKernel(kernels.double_slit_kernel_source,
                                      "double_slit_kernel",
                                      enable_cooperative_groups=False)

    shape = V.shape
    grid_size = math_utils.get_grid_size(shape)
    block_size = (shape[0] // grid_size[0], shape[1] // grid_size[1], shape[2] // grid_size[2])
    double_slit_kernel(
        grid_size,
        block_size,
        (
            V,
            cp.float32(delta_x_3[0]),
            cp.float32(delta_x_3[1]),
            cp.float32(delta_x_3[2]),

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


def particle_hard_interaction_potential(V: cp.ndarray, delta_x_3: np.array, particle_radius_bohr_radius: float,
                                        potential_hartree: float):
    for x in range(V.shape[0]):
        for y in range(V.shape[1]):
            for z in range(V.shape[2]):
                r = math_utils.transform_corner_origin_to_center_origin_system(
                    np.array([x, y, z]) * delta_x_3, delta_x_3 * V.shape
                )
                if (
                        abs(r[0] - r[1]) < 2.0 * particle_radius_bohr_radius
                        or abs(r[1] - r[2]) < 2.0 * particle_radius_bohr_radius
                        or abs(r[2] - r[0]) < 2.0 * particle_radius_bohr_radius
                ):
                    V[x, y, z] += potential_hartree
    return V


def particle_inv_square_interaction_potential(V: cp.ndarray, delta_x_3: np.array, potential_hartree: float):
    for x in range(V.shape[0]):
        for y in range(V.shape[1]):
            for z in range(V.shape[2]):
                r = math_utils.transform_corner_origin_to_center_origin_system(
                    np.array([x, y, z]) * delta_x_3, delta_x_3 * V.shape
                )
                V[x, y, z] += (potential_hartree *
                               (1.0 / max(abs(r[0] - r[1]), 0.0000001) + 1.0 / max(abs(r[0] - r[2]),
                                                                                   0.0000001) + 1.0 / max(
                                   abs(r[1] - r[2]), 0.0000001)))
    return V


def particle_square_interaction_potential(V: cp.ndarray, delta_x_3: np.array, potential_hartree: float):
    for x in range(V.shape[0]):
        for y in range(V.shape[1]):
            for z in range(V.shape[2]):
                r = math_utils.transform_corner_origin_to_center_origin_system(
                    np.array([x, y, z]) * delta_x_3, delta_x_3 * V.shape
                )
                V[x, y, z] += (potential_hartree * (abs(r[0] - r[1]) + abs(r[0] - r[2]) + abs(r[1] - r[2])))
    return V


def add_harmonic_oscillator_for_1D(V: cp.ndarray, delta_x_3: np.array, angular_frequency: float):
    const = 0.5 * math_utils.electron_rest_mass * angular_frequency ** 2
    for x in range(V.shape[0]):
        for y in range(V.shape[1]):
            for z in range(V.shape[2]):
                r = math_utils.transform_corner_origin_to_center_origin_system(
                    np.array([x, y, z]) * delta_x_3, delta_x_3 * V.shape
                )
                V[x, y, z] += const * (r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
    return V


def add_wall_for_1D(V: cp.ndarray, delta_x_3: np.array, center_bohr_radius: float, thickness_bohr_radius: float,
                    potential_hartree: float):
    for x in range(V.shape[0]):
        for y in range(V.shape[1]):
            for z in range(V.shape[2]):
                r = math_utils.transform_corner_origin_to_center_origin_system(
                    np.array([x, y, z]) * delta_x_3, delta_x_3 * V.shape
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
    return V


def add_coulomb_potential(V: cp.ndarray, delta_x_3: np.array, center_bohr_radius: np.array, gradient_dir: np.array,
                          charge_density: float,
                          oxide_start_bohr_radii: float, oxide_end_bohr_radii: float):
    for x in range(V.shape[0]):
        for y in range(V.shape[1]):
            for z in range(V.shape[2]):
                r = math_utils.transform_corner_origin_to_center_origin_system(
                    np.array([x, y, z]) * delta_x_3, delta_x_3 * V.shape
                )
                d = np.dot(gradient_dir, center_bohr_radius - r)
                epsilon = 1.0
                if (d >= oxide_start_bohr_radii and d <= oxide_end_bohr_radii):
                    V[x, y, z] += charge_density / epsilon / 2.0 / max(d,
                                                                       0.00000001) - charge_density / epsilon / 2.0 / oxide_start_bohr_radii
                elif (d > oxide_end_bohr_radii):
                    V[
                        x, y, z] += charge_density / epsilon / 2.0 / oxide_end_bohr_radii - charge_density / epsilon / 2.0 / oxide_start_bohr_radii
    return V


def add_linear_potential_gradient(V: cp.ndarray, delta_x_3: np.array, center_bohr_radius: np.array,
                                  gradient_dir: np.array, gradient_val: float):
    for x in range(V.shape[0]):
        for y in range(V.shape[1]):
            for z in range(V.shape[2]):
                r = math_utils.transform_corner_origin_to_center_origin_system(
                    np.array([x, y, z]) * delta_x_3, delta_x_3 * V.shape
                )
                V[x, y, z] -= gradient_val * abs(np.dot(gradient_dir, r - center_bohr_radius))
    return V


def add_optical_grid(V: cp.ndarray, delta_x_3: np.array, center_bohr_radius: np.array, normal: np.array,
                     distance_between_nodes_bohr_radii: float, potential_hartree: float, node_count: int):
    for x in range(V.shape[0]):
        for y in range(V.shape[1]):
            for z in range(V.shape[2]):
                r = math_utils.transform_corner_origin_to_center_origin_system(
                    np.array([x, y, z]) * delta_x_3, delta_x_3 * V.shape
                )
                right = np.cross(normal, math_utils.prefered_up())
                up = np.cross(right, normal)
                if (abs(np.dot(normal, r - center_bohr_radius)) < 10.0
                        and abs(np.dot(right,
                                       r - center_bohr_radius)) < distance_between_nodes_bohr_radii * node_count * 0.6
                        and abs(
                            np.dot(up, r - center_bohr_radius)) < distance_between_nodes_bohr_radii * node_count * 0.6
                ):
                    for u in range(node_count):
                        for v in range(node_count):
                            current_center = (
                                    center_bohr_radius
                                    + distance_between_nodes_bohr_radii
                                    * (right * float(u - float(node_count) * 0.5) + up * float(
                                v - float(node_count) * 0.5))
                            )

                            d = math_utils.vector_length(current_center - r)
                            V[x, y, z] += potential_hartree * math.exp(-0.5 * d ** 2)
    return V


def generate_potential_from_walls_and_drain(V: cp.ndarray, V_vis: cp.ndarray, delta_x_3: np.array,
                                            absorbing_pontetial: AbsorbingBoundaryCondition, walls: list[PotentialWall]):
    V.fill(0.0)
    for w in walls:
        w.add_potential(
            V=V,
            delta_x_3=delta_x_3,
        )
    #V_vis = V.copy()
    absorbing_pontetial.add_potential(
        V,
        delta_x_3=delta_x_3,
    )

    return V, V_vis
