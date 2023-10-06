import numpy as np
import time
import toml
from sources import wave_packet, potential, operators
import sources.sim_state as sim_st
import sources.text_writer as text_writer
import os
import filecmp
from colorama import Fore, Style


def initialize():
    # We use hartree atomic unit system
    initialisation_start_time_s = time.time()
    use_cache = True
    try:
        with open("config/parameters.toml", mode="r") as f:
            config = toml.load(f)
            try:
                with open("cache/cached_parameters.toml", mode="r") as cache_f:
                    cached_config = toml.load(cache_f)
                    if not cached_config == config:
                        print(
                            "Changes detected in 'parameters.toml'.\n"
                            "Falling back to full initialisation."
                        )
                        use_cache = False
            except OSError as e:
                use_cache = False
            try:
                with open("cache/cached_parameters.toml", mode="w") as cache_f:
                    toml.dump(config, cache_f)
            except OSError as e:
                print("Error while creating parameter cache: " + e)
    except OSError as e:
        print(
            Fore.RED + "No 'config/parameters.toml' found!" + Style.RESET_ALL + "\n"
            "Exiting application."
        )
    sim_state = sim_st.SimState(config)
    sim_state.use_cache = use_cache

    # Maximal kinetic energy
    print(text_writer.get_sim_state_description_text(sim_state, use_colors=True))
    print(
        "***************************************************************************************"
    )

    print("Initializing wave packet")
    sim_state.wp_width_bohr_radii = sim_state.config["Wave_packet"][
        "wp_width_bohr_radii"
    ]
    print(f"Wave packet width is {sim_state.wp_width_bohr_radii} bohr radii.")
    a = sim_state.wp_width_bohr_radii * 2.0

    full_init = True
    if use_cache:
        try:
            sim_state.wave_tensor = np.load(file="cache/gaussian_wave_packet.npy")
            full_init = False
        except OSError:
            print("No cached gaussian_wave_packet.npy found.")

    if full_init:
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

    full_init = True
    if use_cache:
        try:
            sim_state.kinetic_operator = np.load(file="cache/kinetic_operator.npy")
            full_init = False
        except OSError:
            print("No cached kinetic_operator.npy found.")
    if full_init:
        sim_state.kinetic_operator = operators.init_kinetic_operator(
            N=sim_state.N,
            delta_x=sim_state.delta_x_bohr_radii,
            delta_time=sim_state.delta_time_h_bar_per_hartree,
        )
        np.save(file="cache/kinetic_operator.npy", arr=sim_state.kinetic_operator)

    print("Initializing potential energy operator")
    print(text_writer.get_potential_description_text(sim_state, use_colors=True))

    full_init = True
    if use_cache:
        try:
            sim_state.localised_potential_hartree = np.load(
                file="cache/localized_potential.npy"
            )
            full_init = False
        except OSError:
            print("No cached localized_potential.npy found.")
    if full_init:
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

    full_init = True
    if use_cache:
        try:
            sim_state.potential_operator = np.load(file="cache/potential_operator.npy")
            full_init = False
        except OSError:
            print("No cached potential_operator.npy found.")
    if full_init:
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
