import numpy as np
import time
import toml
from sources import wave_packet, potential, operators
import sources.sim_state as sim_st
import sources.text_writer as text_writer
import os


def initialize():
    # We use hartree atomic unit system
    initialisation_start_time_s = time.time()
    with open("config/parameters.toml") as f:
        config = toml.load(f)

    sim_state = sim_st.SimState(config)

    # Maximal kinetic energy
    print(text_writer.get_sim_state_description_text(sim_state))
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
    print(text_writer.get_potential_description_text(sim_state))
    try:
        sim_state.localised_potential_hartree = np.load(
            file="cache/localized_potential.npy"
        )
    except OSError:
        print("No cached localized_potential.npy found.")

        space_between_slits = sim_state.config["Potential"][
            "distance_between_slits_bohr_radii"
        ]
        sim_state.localised_potential_hartree = np.zeros(
            shape=sim_state.tensor_shape, dtype=np.complex_
        )
        sim_state.localised_potential_hartree = potential.add_double_slit(
            delta_x=sim_state.delta_x_bohr_radii,
            center_bohr_radii=np.array([0.0, 0.0, 0.0]),
            thickness_bohr_radii=sim_state.config["Potential"][
                "wall_thickness_bohr_radii"
            ],
            height_hartree=sim_state.config["Potential"]["wall_potential_hartree"],
            slit_width_bohr_radii=sim_state.config["Potential"][
                "slit_width_bohr_radii"
            ],
            shape=sim_state.tensor_shape,
            space_between_slits_bohr_radii=space_between_slits,
            V=sim_state.localised_potential_hartree,
        )
        dp = sim_state.drain_potential_description
        sim_state.localised_potential_hartree = potential.add_draining_potential(
            N=sim_state.N,
            delta_x=sim_state.delta_x_bohr_radii,
            inner_radius_bohr_radii=dp.inner_radius_bohr_radii,
            outer_radius_bohr_radii=dp.outer_radius_bohr_radii,
            max_potential_hartree=dp.max_potential_hartree,
            exponent=dp.exponent,
            V=sim_state.localised_potential_hartree,
        )
        np.save(
            file="cache/localized_potential.npy",
            arr=sim_state.localised_potential_hartree,
        )

    try:
        sim_state.potential_operator = np.load(file="cache/potential_operator.npy")
    except OSError:
        print("No cached potential_operator.npy found.")
        sim_state.potential_operator = operators.init_potential_operator(
            V=sim_state.localised_potential_hartree,
            N=sim_state.N,
            delta_time=sim_state.delta_time_h_bar_per_hartree,
        )
        np.save(file="cache/potential_operator.npy", arr=sim_state.potential_operator)
    print(
        f"Time spent with initialisation: {time.time() - initialisation_start_time_s} s."
    )

    return sim_state
