from typing import Dict
import cupy as cp
import numpy as np
import sources.math_utils as math_utils
from sources import potential


class SimState:
    config: Dict
    tensor_shape: cp.shape
    initial_wp_velocity_bohr_radii_hartree_per_h_bar = np.array([0.0, 0.0, 0.0])
    initial_wp_momentum_h_per_bohr_radius = np.array([0.0, 0.0, 0.0])
    wp_width_bohr_radii = 1.0
    particle_mass = 1.0
    initial_wp_position_bohr_radii_3 = np.array([0.0, 0.0, 0.0])
    drain_potential_description: potential.DrainPotentialDescription
    N = 128
    viewing_window_bottom_corner_voxel: np.array
    viewing_window_top_corner_voxel: np.array
    viewing_window_bottom_corner_bohr_radii: np.array
    viewing_window_top_corner_bohr_radii: np.array
    de_broglie_wave_length_bohr_radii: float
    simulated_volume_width_bohr_radii: float
    delta_x_bohr_radii: float
    upper_limit_on_delta_time_h_per_hartree: float
    delta_time_h_bar_per_hartree: float
    wave_tensor: cp.ndarray
    kinetic_operator: cp.ndarray
    potential_operator: cp.ndarray
    probability_density: np.ndarray
    localised_potential_hartree: np.ndarray
    localised_potential_to_visualize_hartree: np.ndarray
    coulomb_potential: np.ndarray
    use_cache = True
    output_dir: str = ""
    cache_dir: str = ""
    enable_visual_output: bool = True
    simulation_method: str = "fft"
    double_precision_wave_tensor: bool = False
    enable_wave_function_save: bool = True

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
        self.simulated_volume_width_bohr_radii = config["volume"][
            "simulated_volume_width_bohr_radii"
        ]
        self.initial_wp_position_bohr_radii_3 = np.array(
            config["wave_packet"]["initial_wp_position_bohr_radii_3"]
        )
        self.N = config["volume"]["number_of_samples_per_axis"]
        self.tensor_shape = (self.N, self.N, self.N)
        self.coulomb_potential = np.zeros(shape=self.tensor_shape)
        self.delta_x_bohr_radii = self.simulated_volume_width_bohr_radii / self.N
        self.upper_limit_on_delta_time_h_per_hartree = (
            4.0
            / np.pi
            * (3.0 * self.delta_x_bohr_radii * self.delta_x_bohr_radii)
            / 3.0
        )  # Based on reasoning from the Web-Schrödinger paper
        self.delta_time_h_bar_per_hartree = config["iteration"][
            "delta_time_h_bar_per_hartree"
        ]

        # Init draining potential
        self.drain_potential_description = potential.DrainPotentialDescription(config)
        self.viewing_window_bottom_corner_bohr_radii = (
            self.drain_potential_description.boundary_bottom_corner_bohr_radii
        )
        self.viewing_window_top_corner_bohr_radii = (
            self.drain_potential_description.boundary_top_corner_bohr_radii
        )
        self.viewing_window_bottom_corner_voxel = np.array(
            (
                    self.drain_potential_description.boundary_bottom_corner_bohr_radii
                    + np.array([self.simulated_volume_width_bohr_radii, self.simulated_volume_width_bohr_radii, self.simulated_volume_width_bohr_radii])
                    * 0.5
            )
            / self.delta_x_bohr_radii,
            dtype=np.int32,
        )
        self.viewing_window_top_corner_voxel = np.array(
            math_utils.transform_center_origin_to_corner_origin_system(
                self.drain_potential_description.boundary_top_corner_bohr_radii,
                self.simulated_volume_width_bohr_radii,
            )
            / self.delta_x_bohr_radii,
            dtype=np.int32,
        )
        # Swap coordinates if needed
        for i in range(0, 3):
            if (
                self.viewing_window_bottom_corner_voxel[i]
                > self.viewing_window_top_corner_voxel[i]
            ):
                temp = self.viewing_window_bottom_corner_voxel[i]
                self.viewing_window_bottom_corner_voxel[
                    i
                ] = self.viewing_window_top_corner_voxel[i]
                self.viewing_window_top_corner_voxel[i] = temp

        try:
            self.enable_visual_output = config["view"]["enable_visual_output"]
        except KeyError:
            self.enable_visual_output = True
        try:
            self.enable_wave_function_save = config["iteration"]["enable_wave_function_save"]
        except KeyError:
            self.enable_wave_function_save = True


    def get_view_into_raw_wave_function(self):
        return math_utils.cut_window(
            arr=self.wave_tensor,
            bottom=self.viewing_window_bottom_corner_voxel,
            top=self.viewing_window_top_corner_voxel,
        )


    def get_view_into_probability_density(self):
        return math_utils.cut_window(
            arr=self.probability_density,
            bottom=self.viewing_window_bottom_corner_voxel,
            top=self.viewing_window_top_corner_voxel,
        )

    def get_view_into_potential(self):
        return math_utils.cut_window(
            arr=np.real(self.localised_potential_to_visualize_hartree),
            bottom=self.viewing_window_bottom_corner_voxel,
            top=self.viewing_window_top_corner_voxel,
        )

    def get_view_into_coulomb_potential(self):
        return math_utils.cut_window(
            arr=np.real(self.coulomb_potential),
            bottom=self.viewing_window_bottom_corner_voxel,
            top=self.viewing_window_top_corner_voxel,
        )
