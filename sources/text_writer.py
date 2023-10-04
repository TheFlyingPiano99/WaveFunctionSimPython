import sources.sim_state as sim_st
import sources.core_sim as core_sim
import numpy as np
import io
from datetime import timedelta


def get_potential_description_text(sim_state: sim_st.SimState):
    text = io.StringIO()
    text.write("Localised potential energy:\n")
    wall_potential = sim_state.config["Potential"]["wall_potential_hartree"]
    time_times_potential = wall_potential * sim_state.delta_time_h_bar_per_hartree
    text.write(f"Obstacle wall potential is {wall_potential} hartree.\n")
    if abs(time_times_potential / np.pi - time_times_potential // np.pi) < 0.1:
        text.write(
            "WARNING: delta_t * wall_max_potential too close to multiply of pi!\n"
        )
    if time_times_potential > np.pi:
        text.write(
            f"WARNING: delta_t * wall_max_potential = {time_times_potential} exceeds  pi!\n"
        )

    space_between_slits = sim_state.config["Potential"][
        "distance_between_slits_bohr_radii"
    ]
    if space_between_slits > sim_state.de_broglie_wave_length_bohr_radii:
        text.write(
            f"WARNING: Space between slits = {space_between_slits} esceeds de Brogile wavelength = {sim_state.de_broglie_wave_length_bohr_radii} of the particle!"
        )
    drain_max = sim_state.config["Potential"]["outer_drain_potential_hartree"]
    text.write(
        f"Draining potential value at the outer edge is {drain_max:.1f} hartree.\n"
    )
    exponent = sim_state.config["Potential"]["drain_interpolation_exponent"]
    text.write(f"Draining potential exponent is {exponent:.1f}.\n")
    return text.getvalue()


def get_sim_state_description_text(sim_state: sim_st.SimState):
    text = io.StringIO()

    velocity_magnitude = (
        np.dot(
            sim_state.initial_wp_velocity_bohr_radii_hartree_per_h_bar,
            sim_state.initial_wp_velocity_bohr_radii_hartree_per_h_bar,
        )
        ** 0.5
    )
    text.write(
        f"Mass of the particle is {sim_state.particle_mass} electron rest mass.\n"
        f"Initial velocity of the particle is {velocity_magnitude} Bohr radius hartree / h-bar\n"
    )

    momentum_magnitude = (
        np.dot(
            sim_state.initial_wp_momentum_h_per_bohr_radius,
            sim_state.initial_wp_momentum_h_per_bohr_radius,
        )
        ** 0.5
    )
    text.write(
        f"Initial mean momentum of particle is {momentum_magnitude} h-bar / Bohr radius\n"
    )
    text.write(
        f"De Broglie wavelength associated with the particle is {sim_state.de_broglie_wave_length_bohr_radii} Bohr radii.\n"
    )

    initial_kinetic_energy_hartree = (
        momentum_magnitude**2 / 2 / sim_state.particle_mass
    )
    text.write(
        f"Initial mean kinetic energy of the particle is {initial_kinetic_energy_hartree} hartree.\n"
    )

    text.write(
        f"Width of simulated volume is w = {sim_state.simulated_volume_width_bohr_radii} Bohr radii.\n"
    )

    text.write(f"Number of samples per axis is N = {sim_state.N}.\n")

    # Space resolution
    text.write(
        f"Space resolution is delta_x = {sim_state.delta_x_bohr_radii} Bohr radii.\n"
    )
    if (
        sim_state.delta_x_bohr_radii
        >= sim_state.de_broglie_wave_length_bohr_radii / 2.0
    ):
        text.write(
            f"WARNING: delta_x = {sim_state.delta_x_bohr_radii} exceeds half of de Broglie wavelength!\n"
        )

    # The maximum allowed delta_time
    text.write(
        f"The maximal viable time resolution < {sim_state.upper_limit_on_delta_time_h_per_hartree} h-bar / hartree\n"
    )

    # Time increment of simulation
    text.write(
        f"Time resolution is delta = {sim_state.delta_time_h_bar_per_hartree} h-bar / hartree.\n"
    )
    if (
        sim_state.delta_time_h_bar_per_hartree
        > 0.5 * sim_state.upper_limit_on_delta_time_h_per_hartree
    ):
        print("WARNING: delta_time exceeds theoretical limit!")

    return text.getvalue()


def write_sim_state(sim_state: sim_st.SimState):
    with open("output/parameters.txt", mode="w") as f:
        f.write(get_sim_state_description_text(sim_state))
        f.write(get_potential_description_text(sim_state))


def append_iter_data(iter_data: core_sim.IterData):
    with open("output/parameters.txt", mode="a") as f:
        f.write(f"Total iteration count: {iter_data.total_iteration_count}\n")
        f.write(
            f"Total simulated time: {iter_data.total_simulated_time:.4f} h-bar / hartree\n"
        )
        f.write(
            f"Elapsed system time during iteration: {str(timedelta(seconds=iter_data.elapsed_system_time_s))} s\n"
        )
        f.write(
            f"Average system time of an iteration: {iter_data.average_iteration_system_time_s:.4f} s\n"
        )
