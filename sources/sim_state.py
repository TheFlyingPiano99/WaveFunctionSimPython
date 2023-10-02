from typing import Dict
import numpy as np
import sources.math_utils as math_utils
from sources import wave_packet, potential, operators, sim_state


class SimState:
    config: Dict
    initial_wp_velocity_bohr_radii_hartree_per_h_bar = np.array([0.0, 0.0, 0.0])
    initial_wp_momentum_h_per_bohr_radius = np.array([0.0, 0.0, 0.0])
    wp_width_bohr_radii = 1.0
    particle_mass = 1.0
    initial_wp_position_bohr_radii_3 = np.array([0.0, 0.0, 0.0])
    drain_potential: potential.DrainPotentialData
    N = 128
    de_broglie_wave_length_bohr_radii: float
    simulated_volume_width_bohr_radii: float
    delta_x_bohr_radii: float
    upper_limit_on_delta_time_h_per_hartree: float
    delta_time_h_bar_per_hartree: float
    wave_tensor: np.ndarray
    probability_density: np.ndarray
    kinetic_operator: np.ndarray
    only_the_obstacle_potential: np.ndarray
    V: np.ndarray
    potential_operator: np.ndarray

    def __init__(self, config):
        self.config = config
        self.particle_mass = config["Wave_packet"]["particle_mass"]
        self.initial_wp_velocity_bohr_radii_hartree_per_h_bar = np.array(
            config["Wave_packet"]["initial_wp_velocity_bohr_radii_hartree_per_h_bar"]
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
        self.simulated_volume_width_bohr_radii = config["Volume"][
            "simulated_volume_width_bohr_radii"
        ]
        self.initial_wp_position_bohr_radii_3 = np.array(
            config["Wave_packet"]["initial_wp_position_bohr_radii_3"]
        )
        self.N = config["Volume"]["number_of_samples_per_axis"]
        self.delta_x_bohr_radii = self.simulated_volume_width_bohr_radii / self.N
        self.upper_limit_on_delta_time_h_per_hartree = (
            4.0
            / np.pi
            * (3.0 * self.delta_x_bohr_radii * self.delta_x_bohr_radii)
            / 3.0
        )  # Based on reasoning from the Web-Schrödinger paper
        self.delta_time_h_bar_per_hartree = (
            0.5 * self.upper_limit_on_delta_time_h_per_hartree
        )
        # Init draining potential
        self.drain_potential = potential.DrainPotentialData()
        self.drain_potential.boundary_bottom_corner = (
            np.array(
                config["Volume"]["viewing_window_boundary_bottom_corner_bohr_radii_3"]
            ),
            self.simulated_volume_width_bohr_radii,
        )
        self.drain_potential.boundary_top_corner = (
            np.array(
                self.config["Volume"]["viewing_window_boundary_top_corner_bohr_radii_3"]
            ),
            self.simulated_volume_width_bohr_radii,
        )
