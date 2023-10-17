import cupy as cp
import time

import numpy as np
import toml
from sources import wave_packet, potential, operators
import sources.sim_state as sim_st
import sources.text_writer as text_writer
import os
import filecmp
from colorama import Fore, Style
import sources.math_utils as math_utils


def initialize():
    # We use hartree atomic unit system
    initialisation_start_time_s = time.time()
    use_cache = True
    try:
        with open("config/parameters.toml", mode="r") as f:
            config = toml.load(f)
            try:
                if not os.path.exists("cache/"):
                    os.mkdir("cache/")
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
    sim_state.wp_width_bohr_radii = sim_state.config["wave_packet"][
        "wp_width_bohr_radii"
    ]
    print(f"Wave packet width is {sim_state.wp_width_bohr_radii} bohr radii.")
    a = sim_state.wp_width_bohr_radii * 2.0

    full_init = True
    if use_cache:
        try:
            sim_state.wave_tensor = cp.load(file="cache/gaussian_wave_packet.npy")
            full_init = False
        except OSError:
            print("No cached gaussian_wave_packet.npy found.")

    if full_init:
        sim_state.wave_tensor = cp.asarray(
            wave_packet.init_gaussian_wave_packet(
                sim_state.N,
                sim_state.delta_x_bohr_radii,
                a,
                sim_state.initial_wp_position_bohr_radii_3,
                -sim_state.initial_wp_momentum_h_per_bohr_radius,
                sim_state.tensor_shape
            )
        )
        cp.save(file="cache/gaussian_wave_packet.npy", arr=sim_state.wave_tensor)
    # Normalize:
    sim_state.probability_density = cp.asnumpy(cp.square(cp.abs(sim_state.wave_tensor)))
    sum_probability = cp.sum(sim_state.probability_density)
    print(f"Sum of probabilities = {sum_probability}")
    sim_state.wave_tensor = sim_state.wave_tensor / (sum_probability**0.5)
    sim_state.probability_density = cp.asnumpy(cp.square(cp.abs(sim_state.wave_tensor)))
    sum_probability = cp.sum(sim_state.probability_density)
    print(f"Sum of probabilities after normalization = {sum_probability}")
    # Operators:
    print("Initializing kinetic energy operator")

    full_init = True
    if use_cache:
        try:
            sim_state.kinetic_operator = cp.asarray(np.load(file="cache/kinetic_operator.npy"))
            full_init = False
        except OSError:
            print("No cached kinetic_operator.npy found.")
    if full_init:
        sim_state.kinetic_operator = cp.asarray(operators.init_kinetic_operator(
            sim_state.N,
            sim_state.delta_x_bohr_radii,
            sim_state.delta_time_h_bar_per_hartree,
            sim_state.tensor_shape
        ))
        cp.save(file="cache/kinetic_operator.npy", arr=sim_state.kinetic_operator)

    print("Initializing potential energy operator")
    print(text_writer.get_potential_description_text(sim_state, use_colors=True))

    full_init = True
    if use_cache:
        try:
            sim_state.localised_potential_hartree = np.load(file="cache/localized_potential.npy")
            sim_state.localised_potential_to_visualize_hartree = np.load(file="cache/localized_potential_to_visualize.npy")
            full_init = False
        except OSError:
            print("No cached localized_potential.npy found.")
    if full_init:
        sim_state.localised_potential_hartree = np.zeros(
            shape=sim_state.tensor_shape, dtype=np.csingle
        )
        print("Creating draining potential.")
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
        try:
            interaction = sim_state.config["particle_interaction"]
            r = interaction["radius_hartree"]
            v = interaction["potential_hartree"]
            print("Creating particle interaction potential.")
            sim_state.localised_potential_hartree = potential.particle_interaction_potential(
                delta_x=sim_state.delta_x_bohr_radii,
                particle_radius_bohr_radius=r,
                potential_hartree=v,
                V=sim_state.localised_potential_hartree,
            )
        except e:
            pass
        try:
            oscillator = sim_state.config["harmonic_oscillator_1d"]
            omega = oscillator["angular_frequency"]
            w = oscillator["width_bohr_radii"]
            print("Creating harmonic oscillator.")
            sim_state.localised_potential_to_visualize_hartree = potential.add_harmonic_oscillator_for_1D(
                delta_x=sim_state.delta_x_bohr_radii,
                angular_frequency=omega,
                V=np.zeros(shape=sim_state.tensor_shape, dtype=np.csingle),
            )
            sim_state.localised_potential_hartree += sim_state.localised_potential_to_visualize_hartree
        except:
            pass
        for wall_1d in sim_state.config["walls_1d"]:
            c = wall_1d["center_hartree"]
            v = wall_1d["potential_hartree"]
            t = wall_1d["thickness_bohr_radii"]
            print("Creating wall.")
            v_wall = potential.add_wall_for_1D(
                delta_x=sim_state.delta_x_bohr_radii,
                potential_hartree=v,
                V=np.zeros(shape=sim_state.tensor_shape, dtype=np.csingle),
                thickness_bohr_radius=t,
                center_bohr_radius=c
            )
            sim_state.localised_potential_to_visualize_hartree += v_wall
            sim_state.localised_potential_hartree += v_wall

        for double_slit in sim_state.config["double_slits"]:
            print("Creating double-slit.")
            space_between_slits = double_slit["distance_between_slits_bohr_radii"]
            sim_state.localised_potential_hartree = potential.add_double_slit(
                delta_x=sim_state.delta_x_bohr_radii,
                center_bohr_radii=np.array(double_slit["center_bohr_radius_3"]),
                thickness_bohr_radii=double_slit["thickness_bohr_radii"],
                height_hartree=double_slit["potential_hartree"],
                slit_width_bohr_radii=double_slit["slit_width_bohr_radii"],
                shape=sim_state.tensor_shape,
                space_between_slits_bohr_radii=space_between_slits,
                V=sim_state.localised_potential_hartree,
            )

        np.save(
            file="cache/localized_potential.npy",
            arr=sim_state.localised_potential_hartree,
        )
        np.save(file="cache/localized_potential_to_visualize.npy", arr=sim_state.localised_potential_to_visualize_hartree)

    full_init = True
    if use_cache:
        try:
            sim_state.potential_operator = cp.load(file="cache/potential_operator.npy")
            full_init = False
        except OSError:
            print("No cached potential_operator.npy found.")
    if full_init:
        print("Creating potential operator.")
        sim_state.potential_operator = cp.asarray(operators.init_potential_operator(
            V=sim_state.localised_potential_hartree,
            delta_time=sim_state.delta_time_h_bar_per_hartree,
        ))
        cp.save(file="cache/potential_operator.npy", arr=sim_state.potential_operator)
    try:
        with open("cache/cached_parameters.toml", mode="w") as cache_f:
            toml.dump(config, cache_f)
    except OSError as e:
        print("Error while creating parameter cache: " )
        print(e)

    print(
        f"Time spent with initialisation: {time.time() - initialisation_start_time_s} s."
    )

    return sim_state
