import numpy as np
import cupy as cp
import sources.math_utils as math_utils
from numba import jit
import math
import sources.potential_kernels as kernels

class DrainPotentialDescription:
    boundary_bottom_corner_bohr_radii_3: np.array
    boundary_top_corner_bohr_radii_3: np.array
    ellipsoid_a: float
    ellipsoid_b: float
    ellipsoid_c: float
    inner_ellipsoid_distance: float
    max_potential_hartree: float
    exponent: float


    def ellipsoid(self, v: np.array):
        return pow(v[0], 2) / pow(self.ellipsoid_a, 2) + pow(v[1], 2) / pow(self.ellipsoid_b, 2) + pow(v[2], 2) / pow(self.ellipsoid_c, 2) - 1.0

    def __init__(self, config):
        self.boundary_bottom_corner_bohr_radii_3 = np.array(
            config["volume"]["viewing_window_boundary_bottom_corner_bohr_radii_3"]
        )
        self.boundary_top_corner_bohr_radii_3 = np.array(
            config["volume"]["viewing_window_boundary_top_corner_bohr_radii_3"]
        )
        # Flip coordinates if inverted:
        for i in range(3):
            if (
                    self.boundary_bottom_corner_bohr_radii_3[i]
                    > self.boundary_top_corner_bohr_radii_3[i]
            ):
                temp = self.boundary_bottom_corner_bohr_radii_3[i]
                self.boundary_bottom_corner_bohr_radii_3[i] = self.boundary_top_corner_bohr_radii_3[i]
                self.boundary_top_corner_bohr_radii_3[i] = temp

        simulated_volume_dimensions_3 = np.array(config["volume"]["simulated_volume_dimensions_bohr_radii_3"])

        # Clip boundaries if too big:
        for i in range(3):
            if (
                    self.boundary_bottom_corner_bohr_radii_3[i]
                    < -simulated_volume_dimensions_3[i] * 0.5
            ):
                self.boundary_bottom_corner_bohr_radii_3[i] = -simulated_volume_dimensions_3[i] * 0.5
            if (
                    self.boundary_top_corner_bohr_radii_3[i]
                    > simulated_volume_dimensions_3[i] * 0.5
            ):
                self.boundary_top_corner_bohr_radii_3[i] = simulated_volume_dimensions_3[i] * 0.5

        # Find ellipsoid parameters (a, b, c):
        self.ellipsoid_a = simulated_volume_dimensions_3[0] * 0.5
        self.ellipsoid_b = simulated_volume_dimensions_3[1] * 0.5
        self.ellipsoid_c = simulated_volume_dimensions_3[2] * 0.5

        # Find farthest corner of the visualized boundary box:
        max_dist = 0.0
        max_corner = np.array([0.0, 0.0, 0.0])
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    v = np.array([
                        self.boundary_bottom_corner_bohr_radii_3[0] if i else self.boundary_top_corner_bohr_radii_3[0],
                        self.boundary_bottom_corner_bohr_radii_3[1] if j else self.boundary_top_corner_bohr_radii_3[1],
                        self.boundary_bottom_corner_bohr_radii_3[2] if k else self.boundary_top_corner_bohr_radii_3[2],
                    ])
                    d = math_utils.vector_length(v)
                    if (d > max_dist):
                        max_corner = v
                        max_dist = d

        self.max_potential_hartree = config["drain"][
            "outer_potential_hartree"
        ]
        self.exponent = config["drain"]["interpolation_exponent"]

        self.inner_ellipsoid_distance = self.ellipsoid(max_corner)
        if (self.inner_ellipsoid_distance > 0.0):    # The fartherst corner is outside the ellipsoid
            self.inner_ellipsoid_distance = 0
            self.max_potential_hartree = 0.0
            print("The viewing box is too large to add draining potential!")

        print(f"Drain potential: {self.max_potential_hartree}")
        print(f"Exponent: {self.exponent}")
        print(f"Ellips. a: {self.ellipsoid_a}")
        print(f"Ellips. b: {self.ellipsoid_b}")
        print(f"Ellips. c: {self.ellipsoid_c}")
        print(f"Ellips. inner dist.: {self.inner_ellipsoid_distance}")



def add_potential_box(
        voxels_3, delta_x_3, wall_thickness_bohr_radii, potential_wall_height_hartree, V: cp.ndarray = None
):
    if V is None:
        V = cp.zeros(shape=voxels_3, dtype=cp.csingle)
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


def add_draining_potential(
        V: cp.ndarray,
        delta_x_3: np.array,
        ellipsoid_a: float,
        ellipsoid_b: float,
        ellipsoid_c: float,
        inner_ellipsoid_distance: float,
        max_potential_hartree: float,
        exponent: float,
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
            cp.float32(delta_x_3[0]),
            cp.float32(delta_x_3[1]),
            cp.float32(delta_x_3[2]),

            cp.float32(ellipsoid_a),
            cp.float32(ellipsoid_b),
            cp.float32(ellipsoid_c),

            cp.float32(inner_ellipsoid_distance),

            cp.float32(max_potential_hartree),
            cp.float32(exponent),
        )
    )

    return V


def init_potential_sphere(N: int, delta_x_3: np.array, wall_thickness: float, potential_wall_hight: float):
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


def init_zero_potential(N: int):
    V = cp.zeros(shape=(N, N, N), dtype=cp.csingle)
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
    grid_size = (64, 64, 64)
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


def add_wall(
        V: cp.ndarray,
        delta_x_3: np.array,
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
            cp.float32(delta_x_3[0]),
            cp.float32(delta_x_3[1]),
            cp.float32(delta_x_3[2]),

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
                                            drain_description: DrainPotentialDescription, walls: []):
    V.fill(0.0)
    for w in walls:
        add_wall(
            V=V,
            delta_x_3=delta_x_3,
            center_bohr_radii_3=w.center_bohr_radii_3,
            normal_bohr_radii_3=w.normal_bohr_radii_3,
            thickness_bohr_radius=w.thickness_bohr_radii,
            potential_hartree=w.potential_hartree
        )
    V_vis = V.copy()
    add_draining_potential(V=V,
                           delta_x_3=delta_x_3,
                           ellipsoid_a=drain_description.ellipsoid_a,
                           ellipsoid_b=drain_description.ellipsoid_b,
                           ellipsoid_c=drain_description.ellipsoid_c,
                           inner_ellipsoid_distance=drain_description.inner_ellipsoid_distance,
                           max_potential_hartree=drain_description.max_potential_hartree,
                           exponent=drain_description.exponent
                           )

    return V, V_vis
