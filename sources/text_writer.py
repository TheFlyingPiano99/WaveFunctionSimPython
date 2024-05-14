import sources.sim_state as sim_st
import sources.core_sim as core_sim
import numpy as np
import io
from datetime import timedelta
from colorama import Fore, Back, Style
import cupy as cp
import os

def get_title_text():
    text = io.StringIO()
    text.write(Fore.GREEN + "\n ____________________________\n")
    text.write("|                            |\n")
    text.write("|   Wave packet simulation   |\n")
    text.write("|____________________________|\n" + Style.RESET_ALL)
    return text.getvalue()


def write_potential_wall_warnings(text, wall_potential, wall, sim_state, use_colors):
    time_times_potential = wall_potential * sim_state.delta_time_h_bar_per_hartree
    text.write(f"Obstacle wall potential is {wall_potential} Hartree.\n")
    if abs((time_times_potential / cp.pi) - int(time_times_potential / cp.pi)) < 0.05:
        text.write(
            (Fore.RED if use_colors else "")
            + f"WARNING: delta_t * wall_max_potential too close to multiply of pi! ({time_times_potential})\n"
            + (Style.RESET_ALL if use_colors else "")
        )
    thickness = wall["thickness_bohr_radii"]
    text.write(f"Wall thickness is {thickness} Bohr radii.\n")
    if thickness < sim_state.de_broglie_wave_length_bohr_radii:
        text.write(
            "This is thinner than the de Broglie wavelength of the particle.\n"
        )
    else:
        text.write(
            "This is thicker than the de Broglie wavelength of the particle.\n"
        )
    if time_times_potential > cp.pi:
        text.write(
            (Fore.RED if use_colors else "")
            + f"WARNING: delta_t * wall_max_potential = {time_times_potential} exceeds  pi!\n"
            + (Style.RESET_ALL if use_colors else "")
        )

def get_potential_description_text(sim_state: sim_st.SimState, use_colors=False):
    text = io.StringIO()
    text.write(
        (Fore.GREEN if use_colors else "")
        + "Localised potential energy details:\n"
        + (Style.RESET_ALL if use_colors else "")
    )

    try:
        slits = sim_state.config["double_slits"]
        text.write(
            "Double slits:\n"
        )
        for slit in slits:
            wall_potential = slit["potential_hartree"]
            write_potential_wall_warnings(text, wall_potential, slit, sim_state, use_colors)

            space_between_slits = slit["distance_between_slits_bohr_radii"]
            if space_between_slits > sim_state.de_broglie_wave_length_bohr_radii:
                text.write(
                    (Fore.RED if use_colors else "")
                    + f"WARNING: Space between slits = {space_between_slits} exceeds the de Brogile wavelength = {sim_state.de_broglie_wave_length_bohr_radii:.4f} of the particle!"
                    + (Style.RESET_ALL if use_colors else "")
                )
            text.write("\n")
    except KeyError:
        pass

    try:
        walls = sim_state.config["walls"]
        text.write(
            "Walls:\n"
        )
        for i, wall in enumerate(walls):
            text.write(
                f"{i + 1}.\n"
            )
            wall_potential = wall["potential_hartree"]
            write_potential_wall_warnings(text, wall_potential, wall, sim_state, use_colors)
            thickness = wall["thickness_bohr_radii"]
            text.write(f"Wall thickness is {thickness:.2f} Bohr radii.\n")
            text.write("\n")
    except KeyError:
        pass

    try:
        walls = sim_state.config["walls_1d"]
        text.write(
            "1D walls:\n"
        )
        for i, wall in enumerate(walls):
            text.write(
                f"{i + 1}.\n"
            )
            wall_potential = wall["potential_hartree"]
            write_potential_wall_warnings(text, wall_potential, wall, sim_state, use_colors)
            thickness = wall["thickness_bohr_radii"]
            text.write(f"Wall thickness is {thickness:.2f} Bohr radii.\n")
            center = wall["center_bohr_radii"]
            text.write(f"Wall center is at the x = {center:.2f} Bohr radius coordinate.\n")
            text.write("\n")
    except KeyError:
        pass

    try:
        interaction = sim_state.config["particle_hard_interaction"]
        text.write("1D particle hard interaction potential:\n")
        v = interaction["potential_hartree"]
        text.write(f"Potential is {v} Hartree.\n")
        r = interaction["particle_radius_bohr_radii"]
        text.write(f"Particle radius is {r} Bohr radii.\n")
        text.write("\n")
    except KeyError:
        pass

    try:
        interaction = sim_state.config["particle_inv_squared_interaction"]
        text.write("1D particle inverse squared interaction potential:\n")
        v = interaction["central_potential_hartree"]
        text.write(f"Potential at unit distance is {v} Hartree.\n")
        text.write("\n")
    except KeyError:
        pass


    try:
        oscillator = sim_state.config["harmonic_oscillator_1d"]
        text.write("1D particle harmonic oscillator:\n")
        omega = oscillator["angular_frequency"]
        text.write(f"Angular velocity is {omega} radian * Hartree / h-bar.\n")
        text.write("\n")
    except KeyError:
        pass

    drain_max = sim_state.config["drain"]["outer_potential_hartree"]
    text.write(
        f"Draining potential value at the outer edge is {drain_max:.1f} Hartree.\n"
    )
    exponent = sim_state.config["drain"]["interpolation_exponent"]
    text.write(f"Draining potential exponent is {exponent:.1f}.\n")
    return text.getvalue()


def get_sim_state_description_text(sim_state: sim_st.SimState, use_colors=False):
    text = io.StringIO()
    text.write(
        (Fore.GREEN if use_colors else "")
        + "Simulated system state:\n"
        + (Style.RESET_ALL if use_colors else "")
    )
    velocity_magnitude = (
        cp.dot(
            sim_state.initial_wp_velocity_bohr_radii_hartree_per_h_bar,
            sim_state.initial_wp_velocity_bohr_radii_hartree_per_h_bar,
        )
        ** 0.5
    )
    text.write(
        f"Mass of the particle is {sim_state.particle_mass} electron rest mass.\n"
        f"Initial velocity of the particle is {velocity_magnitude} Bohr radius Hartree / h-bar.\n"
    )

    momentum_magnitude = (
        cp.dot(
            sim_state.initial_wp_momentum_h_per_bohr_radius,
            sim_state.initial_wp_momentum_h_per_bohr_radius,
        )
        ** 0.5
    )
    text.write(
        f"Initial mean momentum of particle is {momentum_magnitude} h-bar / Bohr radius.\n"
    )
    text.write(
        f"De Broglie wavelength associated with the particle is {sim_state.de_broglie_wave_length_bohr_radii:.4f} Bohr radii.\n"
    )

    initial_kinetic_energy_hartree = (
        momentum_magnitude**2 / 2 / sim_state.particle_mass
    )
    text.write(
        f"Initial mean kinetic energy of the particle is {initial_kinetic_energy_hartree} Hartree.\n"
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
            (Fore.RED if use_colors else "")
            + f"WARNING: delta_x = {sim_state.delta_x_bohr_radii} exceeds half of de Broglie wavelength!\n"
            + (Style.RESET_ALL if use_colors else "")
        )

    # The maximum allowed delta_time
    text.write(
        f"The maximal viable time resolution < {sim_state.upper_limit_on_delta_time_h_per_hartree} h-bar / Hartree\n"
    )

    # Time increment of simulation
    text.write(
        f"Time resolution is delta = {sim_state.delta_time_h_bar_per_hartree} h-bar / Hartree.\n"
    )
    if (
        sim_state.delta_time_h_bar_per_hartree
        > 0.5 * sim_state.upper_limit_on_delta_time_h_per_hartree
    ):
        text.write(
            (Fore.RED if use_colors else "")
            + "WARNING: delta_time exceeds theoretical limit!\n"
            + (Style.RESET_ALL if use_colors else "")
        )
    # Video
    if sim_state.enable_visual_output:
        fps = sim_state.config["view"]["video_frame_rate"]
        text.write(
            f"Video frame rate is {fps} frame/s.\n"
        )

    return text.getvalue()


def get_finish_text(iter_data):
    text = io.StringIO()
    text.write(f"Total iteration count is {iter_data.total_iteration_count}.\n")
    text.write(
        f"Total simulated time is {iter_data.total_simulated_time:.4f} h-bar / Hartree.\n"
    )
    text.write(
        f"Elapsed system time during iteration was {str(timedelta(seconds=iter_data.elapsed_system_time_s))}.\n"
    )
    text.write(
        f"Average system time of an iteration was {iter_data.average_iteration_system_time_s:.4f} s.\n"
    )
    return text.getvalue()


def get_simulation_method_text(sim_state: sim_st.SimState, use_colors=False):
    if sim_state.simulation_method == "fft":
        print((Fore.BLUE if use_colors else "") + "Using the Split-Operator Fourier method to simulate the time development." + (Style.RESET_ALL if use_colors else ""))
    elif sim_state.simulation_method == "power_series":
        print((Fore.BLUE if use_colors else "") + "Using the Power Series method to simulate the time development.")
        print("The order of approximation is p = 10." + (Style.RESET_ALL if use_colors else ""))
    pass


def write_sim_state(sim_state: sim_st.SimState):
    with open(os.path.join(sim_state.output_dir, "parameters.txt"), mode="w") as f:
        f.write(get_sim_state_description_text(sim_state))
        f.write(get_potential_description_text(sim_state))
        f.write(get_simulation_method_text(sim_state))
        f.write(f"Simulation method: {sim_state.simulation_method}")


def append_iter_data(iter_data: core_sim.IterData, sim_state: sim_st.SimState):
    with open(os.path.join(sim_state.output_dir, "parameters.txt"), mode="a") as f:
        f.write(get_finish_text(iter_data))
