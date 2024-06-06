from typing import Dict
import cupy as cp
import numpy as np
import sources.math_utils as math_utils
import sources.operators
from sources import potential


class PotentialWall:
    center_bohr_radii_3: np.array = np.array([0.0, 0.0, 0.0])
    normal_bohr_radii_3: np.array = np.array([1.0, 0.0, 0.0])
    thickness_bohr_radii: float = 5.0
    potential_hartree: float = 20.0
    velocity_bohr_radius_hartree_per_h_bar: np.array = np.array([0.0, 0.0, 0.0])
    angular_velocity_rad_hartree_per_h_bar: np.array = np.array([0.0, 0.0, 0.0])
    potential_change_rate_hartree_2_per_h_bar: float = 0.0

class SimState:
    config: Dict
    initial_wp_velocity_bohr_radii_hartree_per_h_bar = np.array([0.0, 0.0, 0.0])
    initial_wp_momentum_h_per_bohr_radius = np.array([0.0, 0.0, 0.0])
    wp_width_bohr_radii = 1.0
    particle_mass = 1.0
    initial_wp_position_bohr_radii_3 = np.array([0.0, 0.0, 0.0])
    drain_potential_description: potential.DrainPotentialDescription
    number_of_voxels_3 = np.array([128, 128, 128])
    viewing_window_bottom_corner_voxel_3: np.array
    viewing_window_top_corner_voxel_3: np.array
    viewing_window_bottom_corner_bohr_radii_3: np.array
    viewing_window_top_corner_bohr_radii_3: np.array
    de_broglie_wave_length_bohr_radii: float
    simulated_volume_dimensions_bohr_radii_3: np.array([120.0, 120.0, 120.0])
    delta_x_bohr_radii_3: np.array([0.0, 0.0, 0.0])
    upper_limit_on_delta_time_h_per_hartree: float
    delta_time_h_bar_per_hartree: float
    wave_tensor: cp.ndarray
    kinetic_operator: cp.ndarray
    potential_operator: cp.ndarray
    probability_density: cp.ndarray
    localised_potential_hartree: cp.ndarray
    localised_potential_to_visualize_hartree: cp.ndarray
    coulomb_potential: cp.ndarray
    use_cache = True
    output_dir: str = ""
    cache_dir: str = ""
    enable_visual_output: bool = True
    simulation_method: str = "fft"
    double_precision_wave_tensor: bool = False
    enable_wave_function_save: bool = True
    potential_walls = []
    is_dynamic_potential_mode: bool = False

    def __init__(self, config):
        # Load paths:
        try:
            self.cache_dir = config["paths"]["cache_dir"]
        except KeyError:
            self.cache_dir = "cache/"

        try:
            self.output_dir = config["paths"]["output_dir"]
        except KeyError:
            self.output_dir = "output/"

        try:
            self.simulation_method = config["iteration"]["method"]
            if self.simulation_method not in ["fft", "power_series"]:
                self.simulation_method = "fft"
        except KeyError:
            self.simulation_method = "fft"

        try:
            self.double_precision_wave_tensor = config["volume"]["double_precision_wave_tensor"]
        except KeyError:
            self.double_precision_wave_tensor = False

        self.config = config
        self.particle_mass = config["wave_packet"]["particle_mass"]
        self.initial_wp_velocity_bohr_radii_hartree_per_h_bar = np.array(
            config["wave_packet"]["initial_wp_velocity_bohr_radii_hartree_per_h_bar"]
        )
        self.initial_wp_momentum_h_per_bohr_radius = math_utils.classical_momentum(
            mass=self.particle_mass,
            velocity=self.initial_wp_velocity_bohr_radii_hartree_per_h_bar,
        )
        momentum_magnitude = (
            np.dot(
                self.initial_wp_momentum_h_per_bohr_radius,
                self.initial_wp_momentum_h_per_bohr_radius,
            )
            ** 0.5
        )
        self.de_broglie_wave_length_bohr_radii = (
            math_utils.get_de_broglie_wave_length_bohr_radii(momentum_magnitude)
        )
        self.simulated_volume_dimensions_bohr_radii_3 = np.array(config["volume"][
            "simulated_volume_dimensions_bohr_radii_3"
        ])
        self.initial_wp_position_bohr_radii_3 = np.array(
            config["wave_packet"]["initial_wp_position_bohr_radii_3"]
        )
        self.number_of_voxels_3 = config["volume"]["number_of_voxels_3"]
        self.coulomb_potential = cp.zeros(shape=self.number_of_voxels_3)
        self.delta_x_bohr_radii_3 = self.simulated_volume_dimensions_bohr_radii_3 / self.number_of_voxels_3
        self.upper_limit_on_delta_time_h_per_hartree = (
            4.0
            / np.pi
            * (3.0 * np.max(self.delta_x_bohr_radii_3) * np.max(self.delta_x_bohr_radii_3))
            / 3.0
        )  # Based on reasoning from the Web-Schrödinger paper
        self.delta_time_h_bar_per_hartree = config["simulation"][
            "delta_time_h_bar_per_hartree"
        ]

        # Init draining potential
        self.drain_potential_description = potential.DrainPotentialDescription(config)  # It processes the viewing boundaries
        self.viewing_window_bottom_corner_bohr_radii_3 = (
            self.drain_potential_description.boundary_bottom_corner_bohr_radii_3
        )
        self.viewing_window_top_corner_bohr_radii_3 = (
            self.drain_potential_description.boundary_top_corner_bohr_radii_3
        )
        self.viewing_window_bottom_corner_voxel_3 = np.array(
            (
                    self.drain_potential_description.boundary_bottom_corner_bohr_radii_3
                    + self.simulated_volume_dimensions_bohr_radii_3
                    * 0.5
            )
            / self.delta_x_bohr_radii_3,
            dtype=np.int32,
        )
        self.viewing_window_top_corner_voxel_3 = np.array(
            math_utils.transform_center_origin_to_corner_origin_system(
                self.drain_potential_description.boundary_top_corner_bohr_radii_3,
                self.simulated_volume_dimensions_bohr_radii_3,
            )
            / self.delta_x_bohr_radii_3,
            dtype=np.int32,
        )
        # Swap coordinates if needed
        for i in range(3):
            if (
                self.viewing_window_bottom_corner_voxel_3[i]
                > self.viewing_window_top_corner_voxel_3[i]
            ):
                temp = self.viewing_window_bottom_corner_voxel_3[i]
                self.viewing_window_bottom_corner_voxel_3[
                    i
                ] = self.viewing_window_top_corner_voxel_3[i]
                self.viewing_window_top_corner_voxel_3[i] = temp

        # Correct max voxel if out of simulated voxel count:
        for i in range(3):
            if (self.viewing_window_top_corner_voxel_3[i] >= self.number_of_voxels_3[i]):
                self.viewing_window_top_corner_voxel_3[i] = self.number_of_voxels_3[i] - 1

        self.viewing_window_top_corner_voxel_3 -= np.array([0, 1, 1]) # For testing

        print(f"Bottom corner voxel: ({self.viewing_window_bottom_corner_voxel_3[0]}, {self.viewing_window_bottom_corner_voxel_3[1]}, {self.viewing_window_bottom_corner_voxel_3[2]})")
        print(f"Top corner voxel: ({self.viewing_window_top_corner_voxel_3[0]}, {self.viewing_window_top_corner_voxel_3[1]}, {self.viewing_window_top_corner_voxel_3[2]})")


        try:
            self.enable_visual_output = config["view"]["enable_visual_output"]
        except KeyError:
            self.enable_visual_output = True
        try:
            self.enable_wave_function_save = config["view"]["enable_wave_function_save"]
        except KeyError:
            self.enable_wave_function_save = True
        try:
            self.is_dynamic_potential_mode = config["simulation"]["is_dynamic_potential_mode"]
        except KeyError:
            self.is_dynamic_potential_mode = False


    def get_view_into_raw_wave_function(self):
        return math_utils.cut_window(
            arr=self.wave_tensor,
            bottom=self.viewing_window_bottom_corner_voxel_3,
            top=self.viewing_window_top_corner_voxel_3,
        )


    def get_view_into_probability_density(self):
        return math_utils.cut_window(
            arr=self.probability_density,
            bottom=self.viewing_window_bottom_corner_voxel_3,
            top=self.viewing_window_top_corner_voxel_3,
        )

    def get_view_into_potential(self):
        return math_utils.cut_window(
            arr=cp.real(self.localised_potential_to_visualize_hartree),
            bottom=self.viewing_window_bottom_corner_voxel_3,
            top=self.viewing_window_top_corner_voxel_3,
        )

    def get_view_into_complex_potential(self):
        return math_utils.cut_window(
            arr=self.localised_potential_to_visualize_hartree,
            bottom=self.viewing_window_bottom_corner_voxel_3,
            top=self.viewing_window_top_corner_voxel_3,
        )

    def get_view_into_coulomb_potential(self):
        return math_utils.cut_window(
            arr=self.coulomb_potential,
            bottom=self.viewing_window_bottom_corner_voxel_3,
            top=self.viewing_window_top_corner_voxel_3,
        )

    def update_potential(self):
        if not self.is_dynamic_potential_mode:
            return
        for w in self.potential_walls:
            # Advect:
            w.center_bohr_radii_3 = w.center_bohr_radii_3 + w.velocity_bohr_radius_hartree_per_h_bar * self.delta_time_h_bar_per_hartree
            if w.center_bohr_radii_3[0] < 0.0:  # Stop at zero (For testing only)
                w.center_bohr_radii_3[0] = 0.0

            """
            # Rotate around Z axis:
            w.normal_bohr_radii_3 = np.dot(
                math_utils.rotation_matrix(
                    np.array([0, 0, 1]),
                    w.angular_velocity_rad_hartree_per_h_bar[2] * self.delta_time_h_bar_per_hartree
                ),
                w.normal_bohr_radii_3
            )
            # Rotate around Y axis:
            w.normal_bohr_radii_3 = np.dot(
                math_utils.rotation_matrix(
                    np.array([0, 1, 0]),
                    w.angular_velocity_rad_hartree_per_h_bar[1] * self.delta_time_h_bar_per_hartree
                ),
                w.normal_bohr_radii_3
            )
            # Rotate around X axis:
            w.normal_bohr_radii_3 = np.dot(
                math_utils.rotation_matrix(
                    np.array([1, 0, 0]),
                    w.angular_velocity_rad_hartree_per_h_bar[0] * self.delta_time_h_bar_per_hartree
                ),
                w.normal_bohr_radii_3
            )
            w.potential_hartree = w.potential_change_rate_hartree_2_per_h_bar * self.delta_time_h_bar_per_hartree
        """

        self.localised_potential_hartree, self.localised_potential_to_visualize_hartree = potential.generate_potential_from_walls_and_drain(
            V=self.localised_potential_hartree,
            V_vis=self.localised_potential_to_visualize_hartree,
            delta_x_3=self.delta_x_bohr_radii_3,
            drain_description=self.drain_potential_description,
            walls=self.potential_walls
        )

        self.potential_operator = sources.operators.init_potential_operator(
            P_potential=self.potential_operator,
            V=self.localised_potential_hartree,
            delta_time=self.delta_time_h_bar_per_hartree,
        )