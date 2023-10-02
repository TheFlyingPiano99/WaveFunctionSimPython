import numpy as np
import time
import toml
from sources import wave_packet, potential, operators, text_writer
import sources.sim_state as sim_st


def initialize():
    # We use hartree atomic unit system
    initialisation_start_time_s = time.time()
    with open("config/parameters.toml") as f:
        config = toml.load(f)

    sim_state = sim_st.SimState(config)

    # Maximal kinetic energy
    velocity_magnitude = (
        np.dot(
            sim_state.initial_wp_velocity_bohr_radii_hartree_per_h_bar,
            sim_state.initial_wp_velocity_bohr_radii_hartree_per_h_bar,
        )
        ** 0.5
    )
    print(
        f"Mass of the particle is {sim_state.particle_mass} electron rest mass.\n"
        f"Initial velocity of the particle is {velocity_magnitude} Bohr radius hartree / h-bar"
    )

    momentum_magnitude = (
        np.dot(
            sim_state.initial_wp_momentum_h_per_bohr_radius,
            sim_state.initial_wp_momentum_h_per_bohr_radius,
        )
        ** 0.5
    )
    print(
        f"Initial mean momentum of particle is {momentum_magnitude} h-bar / Bohr radius"
    )
    print(
        f"De Broglie wavelength associated with the particle is {sim_state.de_broglie_wave_length_bohr_radii} Bohr radii."
    )

    initial_kinetic_energy_hartree = (
        momentum_magnitude**2 / 2 / sim_state.particle_mass
    )
    print(
        f"Initial mean kinetic energy of the particle is {initial_kinetic_energy_hartree} hartree."
    )

    print(
        f"Width of simulated volume is w = {sim_state.simulated_volume_width_bohr_radii} Bohr radii."
    )

    print(f"Number of samples per axis is N = {sim_state.N}.")

    # Space resolution
    print(f"Space resolution is delta_x = {sim_state.delta_x_bohr_radii} Bohr radii.")
    if (
        sim_state.delta_x_bohr_radii
        >= sim_state.de_broglie_wave_length_bohr_radii / 2.0
    ):
        print("WARNING: delta_x exceeds half of de Broglie wavelength!")

    # The maximum allowed delta_time
    print(
        f"The maximal viable time resolution < {sim_state.upper_limit_on_delta_time_h_per_hartree} h-bar / hartree"
    )

    # Time increment of simulation
    print(
        f"Time resolution is delta = {sim_state.delta_time_h_bar_per_hartree} h-bar / hartree."
    )

    print(
        "***************************************************************************************"
    )

    print("Initializing wave packet")
    sim_state.wp_width_bohr_radii = sim_state.config["Wave_packet"][
        "wp_width_bohr_radii"
    ]
    print(f"Wave packet width is {sim_state.wp_width_bohr_radii} bohr radii.")
    a = sim_state.wp_width_bohr_radii * 2.0
    try:
        sim_state.wave_tensor = np.load(file="cache/gaussian_wave_packet.npy")
    except OSError:
        print("No cached gaussian_wave_packet.npy found.")
        sim_state.wave_tensor = wave_packet.init_gaussian_wave_packet(
            N=sim_state.N,
            delta_x_bohr_radii=sim_state.delta_x_bohr_radii,
            a=a,
            r_0_bohr_radii_3=sim_state.initial_wp_position_bohr_radii_3,
            initial_momentum_h_per_bohr_radius_3=-sim_state.initial_wp_momentum_h_per_bohr_radius,
        )
        np.save(file="cache/gaussian_wave_packet.npy", arr=sim_state.wave_tensor)

    # Normalize:
    sim_state.probability_density = np.square(np.abs(sim_state.wave_tensor))
    sum_probability = np.sum(sim_state.probability_density)
    print(f"Sum of probabilities = {sum_probability}")
    sim_state.wave_tensor = sim_state.wave_tensor / (sum_probability**0.5)
    sim_state.probability_density = np.square(np.abs(sim_state.wave_tensor))
    sum_probability = np.sum(sim_state.probability_density)
    print(f"Sum of probabilities after normalization = {sum_probability}")
    # Operators:
    print("Initializing kinetic energy operator")
    try:
        sim_state.kinetic_operator = np.load(file="cache/kinetic_operator.npy")
    except OSError:
        print("No cached kinetic_operator.npy found.")
        sim_state.kinetic_operator = operators.init_kinetic_operator(
            N=sim_state.N,
            delta_x=sim_state.delta_x_bohr_radii,
            delta_time=sim_state.delta_time_h_bar_per_hartree,
        )
        np.save(file="cache/kinetic_operator.npy", arr=sim_state.kinetic_operator)

    print("Initializing potential energy operator")
    try:
        sim_state.V = np.load(file="cache/localized_potential.npy")
        sim_state.only_the_obstacle_potential = np.load(
            file="cache/only_the_obstacle_potential.npy"
        )
    except OSError:
        print("No cached localized_potential.npy found.")
        sim_state.V = np.zeros(
            shape=(sim_state.N, sim_state.N, sim_state.N), dtype=float
        )
        # V = potential.add_single_slit(V=V, delta_x=delta_x_bohr_radii, center_bohr_radii=15.0, thickness_bohr_radii=1.0, height_hartree=200.0, slit_size_bohr_radii=3.0)
        sim_state.V = potential.add_double_slit(
            V=sim_state.V,
            delta_x=sim_state.delta_x_bohr_radii,
            center_bohr_radii=np.array([0.0, 15.0, 15.0]),
            thickness_bohr_radii=1.5,
            height_hartree=200.0,
            slit_width_bohr_radii=2,
            space_between_slits_bohr_radii=0.5,
        )
        # sim_state.V = potential.add_wall(V=sim_state.V, delta_x=delta_x_bohr_radii, center_bohr_radii=15.0, thickness_bohr_radii=1.5, height_hartree=200)
        sim_state.only_the_obstacle_potential = sim_state.V.copy()
        sim_state.V = potential.add_potential_box(
            V=sim_state.V,
            N=sim_state.N,
            delta_x=sim_state.delta_x_bohr_radii,
            wall_thickness_bohr_radii=1.5,
            potential_wall_height_hartree=1000.0,
        )
        np.save(file="cache/localized_potential.npy", arr=sim_state.V)
        np.save(
            file="cache/only_the_obstacle_potential.npy",
            arr=sim_state.only_the_obstacle_potential,
        )

    try:
        sim_state.potential_operator = np.load(file="cache/potential_operator.npy")
    except OSError:
        print("No cached potential_operator.npy found.")
        sim_state.potential_operator = operators.init_potential_operator(
            V=sim_state.V,
            N=sim_state.N,
            delta_time=sim_state.delta_time_h_bar_per_hartree,
        )
        np.save(file="cache/potential_operator.npy", arr=sim_state.potential_operator)
    print(
        f"Time spent with initialisation: {time.time() - initialisation_start_time_s} s."
    )

    text_writer.write_sim_state(sim_state)
    return sim_state
