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
import sys

def is_toml(file_name):
    return file_name.endswith('.toml')

def initialize():
    initialisation_start_time_s = time.time()
    use_cache = True
    config_dir = "config/"
    if os.path.exists(config_dir):
        conf_files = list(filter(is_toml, os.listdir(config_dir)))
        if len(conf_files) == 1:
            selected_conf_file = conf_files[0]
        elif len(conf_files) > 1:
            answer = -1
            while not answer in range(len(conf_files)):
                print("Available configuration files:")
                for idx, file  in enumerate(conf_files):
                    print(f"{idx} {file}")
                print("Select one by entering its index number:", end=" ")
                answer = int(input())
            selected_conf_file = conf_files[answer]
        else:
            print("No config file found under config folder.")
            print("Exiting application.")
            sys.exit(0)


        # Opening the selected file:
        with open(config_dir + selected_conf_file, mode="r") as f:
            print(f"Opening {selected_conf_file}")
            config = toml.load(f)
            sim_state = sim_st.SimState(config)
            try:
                if not os.path.exists(sim_state.cache_dir):
                    os.makedirs(sim_state.cache_dir, exist_ok=True)
                with open(os.path.join(sim_state.cache_dir, "cached_parameters.toml"), mode="r") as cache_f:
                    cached_config = toml.load(cache_f)
                    if not cached_config == config:
                        print(
                            "Configuration file is different from the one used last time.\n"
                            "Falling back to full initialisation."
                        )
                        use_cache = False
            except OSError as e:
                use_cache = False
    else:
        print("No config folder found.")
        print("Exiting application.")
        sys.exit(0)

    sim_state.use_cache = use_cache


    # Warn user about not empty output directory:
    if not os.path.exists(sim_state.output_dir):
        os.makedirs(sim_state.output_dir, exist_ok=True)
    if os.listdir(sim_state.output_dir):
        answer = ""
        while not answer in {"y", "n"}:
            print("\n" + Fore.RED + "Output directory is not empty!\n"
                'Continuing will possibly override files under "' + sim_state.output_dir + '".' + Style.RESET_ALL + "\n"
                                                                              "Would you still like to continue [y/n]?",
                end=" ",
                )
            answer = input()
            if answer == "n":
                print("Exiting application.")
                sys.exit(0)

    print("") # Empty line

    # Maximal kinetic energy
    print(text_writer.get_sim_state_description_text(sim_state, use_colors=True))
    print(
        "***************************************************************************************"
    )

    print("")
    print(Fore.GREEN + "Initializing wave packet" + Style.RESET_ALL)
    sim_state.wp_width_bohr_radii = sim_state.config["wave_packet"][
        "wp_width_bohr_radii"
    ]
    print(f"Wave packet width is {sim_state.wp_width_bohr_radii} Bohr radii.")
    a = sim_state.wp_width_bohr_radii * 2.0

    full_init = True
    if use_cache:
        try:
            sim_state.wave_tensor = cp.load(file=os.path.join(sim_state.cache_dir, "gaussian_wave_packet.npy"))
            full_init = False
        except OSError:
            print("No cached gaussian_wave_packet.npy found.")

    if full_init:
        sim_state.wave_tensor = cp.asarray(
            wave_packet.init_gaussian_wave_packet_double_precision(
                sim_state.N,
                sim_state.delta_x_bohr_radii,
                a,
                sim_state.initial_wp_position_bohr_radii_3,
                sim_state.initial_wp_momentum_h_per_bohr_radius,
                sim_state.tensor_shape,
            )
            if sim_state.double_precision_wave_tensor else
            wave_packet.init_gaussian_wave_packet_single_precision(
                sim_state.N,
                sim_state.delta_x_bohr_radii,
                a,
                sim_state.initial_wp_position_bohr_radii_3,
                sim_state.initial_wp_momentum_h_per_bohr_radius,
                sim_state.tensor_shape,
            )
        )
        cp.save(file=os.path.join(sim_state.cache_dir, "gaussian_wave_packet.npy"), arr=sim_state.wave_tensor)
    # Normalize:
    sim_state.probability_density = cp.asnumpy(cp.square(cp.abs(sim_state.wave_tensor)))
    sum_probability = cp.sum(sim_state.probability_density)
    print(f"Sum of probabilities = {sum_probability:.8f}")
    sim_state.wave_tensor = sim_state.wave_tensor / (sum_probability**0.5)
    sim_state.probability_density = cp.asnumpy(cp.square(cp.abs(sim_state.wave_tensor)))
    sum_probability = cp.sum(sim_state.probability_density)
    print(f"Sum of probabilities after normalization = {sum_probability:.8f}")

    if sim_state.simulation_method == "fft":
        # Operators:
        print("")
        print(Fore.GREEN + "Initializing kinetic energy operator" + Style.RESET_ALL)

        full_init = True
        if use_cache:
            try:
                sim_state.kinetic_operator = cp.asarray(np.load(file=os.path.join(sim_state.cache_dir, "kinetic_operator.npy")))
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
            cp.save(file=os.path.join(sim_state.cache_dir, "kinetic_operator.npy"), arr=sim_state.kinetic_operator)

        print("")
        print(Fore.GREEN + "Initializing potential energy operator" + Style.RESET_ALL)
        print("")
        print(text_writer.get_potential_description_text(sim_state, use_colors=True))

    full_init = True
    if use_cache:
        try:
            sim_state.localised_potential_hartree = np.load(file=os.path.join(sim_state.cache_dir, "localized_potential.npy"))
            sim_state.localised_potential_to_visualize_hartree = np.load(file=os.path.join(sim_state.cache_dir, "localized_potential_to_visualize.npy"))
            full_init = False
        except OSError:
            print("No cached localized_potential.npy found.")

    if full_init:
        sim_state.localised_potential_hartree = np.zeros(
            shape=sim_state.tensor_shape, dtype=np.complex64
        )
        sim_state.localised_potential_to_visualize_hartree = np.zeros(
            shape=sim_state.tensor_shape, dtype=np.csingle
        )

        # Load pre-initialized potential:
        try:
            pre_init_pot_conf = sim_state.config["pre_initialized_potential"]
            print("Loading pre-initialized potential")
            if os.path.exists(pre_init_pot_conf["path"]):
                try:
                    pre_init_pot = np.load(file=pre_init_pot_conf["path"])
                    if pre_init_pot.shape == sim_state.localised_potential_hartree.shape:
                        sim_state.localised_potential_hartree += pre_init_pot
                        try:
                            visible = pre_init_pot_conf["visible"]
                            if visible:
                                sim_state.localised_potential_to_visualize_hartree += pre_init_pot
                        except KeyError:
                            pass

                    else:
                        print(Fore.RED + "Pre-initialized potential has the wrong tensor shape!" + Style.RESET_ALL)
                except IOError:
                    print(Fore.RED + "Found pre-initialized potential but failed to load!" + Style.RESET_ALL)
            else:
                print(Fore.RED + "Path to pre-initialized potential (" + pre_init_pot_conf["path"] + ") is invalid!" + Style.RESET_ALL)
        except KeyError:
            print("No pre-initialized potential in configuration.")


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
            interaction = sim_state.config["particle_hard_interaction"]
            r = interaction["particle_radius_bohr_radii"]
            v = interaction["potential_hartree"]
            print("Creating particle hard interaction potential.")
            tensor = potential.particle_hard_interaction_potential(
                delta_x=sim_state.delta_x_bohr_radii,
                particle_radius_bohr_radius=r,
                potential_hartree=v,
                V=np.zeros(shape=sim_state.tensor_shape, dtype=np.csingle),
            )
            sim_state.localised_potential_hartree += tensor
            try:
                visible = interaction["visible"]
                if visible:
                    sim_state.localised_potential_to_visualize_hartree += tensor
            except KeyError:
                pass
        except KeyError:
            pass

        try:
            interaction = sim_state.config["particle_inv_squared_interaction"]
            v = interaction["center_potential_hartree"]
            print("Creating particle inverse square interaction potential.")
            tensor = potential.particle_inv_square_interaction_potential(
                delta_x=sim_state.delta_x_bohr_radii,
                potential_hartree=v,
                V=np.zeros(shape=sim_state.tensor_shape, dtype=np.csingle),
            )
            sim_state.localised_potential_hartree += tensor
            try:
                visible = interaction["visible"]
                if visible:
                    sim_state.localised_potential_to_visualize_hartree += tensor
            except KeyError:
                pass
        except KeyError:
            pass

        try:
            oscillator = sim_state.config["harmonic_oscillator_1d"]
            omega = oscillator["angular_frequency_radian_hartree_per_h_bar"]
            print("Creating harmonic oscillator.")
            tensor = potential.add_harmonic_oscillator_for_1D(
                delta_x=sim_state.delta_x_bohr_radii,
                angular_frequency=omega,
                V=np.zeros(shape=sim_state.tensor_shape, dtype=np.csingle),
            )
            sim_state.localised_potential_hartree += tensor
            try:
                visible = oscillator["visible"]
                if visible:
                    sim_state.localised_potential_to_visualize_hartree += tensor
            except KeyError:
                pass
        except KeyError:
            pass

        try:
            walls_arr = sim_state.config["walls_1d"]
            for wall_1d in walls_arr:
                c = wall_1d["center_bohr_radii"]
                v = wall_1d["potential_hartree"]
                t = wall_1d["thickness_bohr_radii"]
                print("Creating wall.")
                tensor = potential.add_wall_for_1D(
                    delta_x=sim_state.delta_x_bohr_radii,
                    potential_hartree=v,
                    thickness_bohr_radius=t,
                    center_bohr_radius=c,
                    V=np.zeros(shape=sim_state.tensor_shape, dtype=np.csingle),
                )
                sim_state.localised_potential_hartree += tensor
                try:
                    visible = wall_1d["visible"]
                    if visible:
                        sim_state.localised_potential_to_visualize_hartree += tensor
                except KeyError:
                    pass
        except KeyError:
            pass

        try:
            walls_arr = sim_state.config["walls"]
            for wall in walls_arr:
                v = wall["potential_hartree"]
                c = np.array(wall["center_bohr_radii_3"], dtype=float)
                n = math_utils.normalize(np.array(wall["normal_vector_3"], dtype=float))
                t = wall["thickness_bohr_radii"]
                print("Creating wall.")
                tensor = potential.add_wall(
                    delta_x=sim_state.delta_x_bohr_radii,
                    potential_hartree=v,
                    thickness_bohr_radius=t,
                    center_bohr_radius=c,
                    normal=n,
                    V=np.zeros(shape=sim_state.tensor_shape, dtype=np.csingle),
                )
                sim_state.localised_potential_hartree += tensor
                try:
                    visible = wall["visible"]
                    if visible:
                        sim_state.localised_potential_to_visualize_hartree += tensor
                except KeyError:
                    pass
        except KeyError:
            pass

        try:
            grid_arr = sim_state.config["optical_grids"]
            for grid in grid_arr:
                v = grid["potential_hartree"]
                c = np.array(grid["center_bohr_radii_3"], dtype=float)
                n = math_utils.normalize(np.array(grid["normal_vector_3"], dtype=float))
                d = grid["distance_between_nodes_bohr_radii"]
                i = grid["node_in_one_direction"]
                print("Creating optical grid.")
                tensor = potential.add_optical_grid(
                    delta_x=sim_state.delta_x_bohr_radii,
                    potential_hartree=v,
                    distance_between_nodes_bohr_radii=d,
                    center_bohr_radius=c,
                    normal=n,
                    node_count=i,
                    V=np.zeros(shape=sim_state.tensor_shape, dtype=np.csingle),
                )
                sim_state.localised_potential_hartree += tensor
                try:
                    visible = grid["visible"]
                    if visible:
                        sim_state.localised_potential_to_visualize_hartree += tensor
                except KeyError:
                    pass
        except KeyError:
            pass

        try:
            ds_array = sim_state.config["double_slits"]
            for double_slit in ds_array:
                print("Creating double-slit.")
                space_between_slits = double_slit["distance_between_slits_bohr_radii"]
                tensor = potential.add_double_slit(
                    delta_x=sim_state.delta_x_bohr_radii,
                    center_bohr_radii=np.array(double_slit["center_bohr_radius_3"]),
                    thickness_bohr_radii=double_slit["thickness_bohr_radii"],
                    height_hartree=double_slit["potential_hartree"],
                    slit_width_bohr_radii=double_slit["slit_width_bohr_radii"],
                    shape=sim_state.tensor_shape,
                    space_between_slits_bohr_radii=space_between_slits,
                    V=np.zeros(shape=sim_state.tensor_shape, dtype=np.csingle),
                )
                sim_state.localised_potential_hartree += tensor
                try:
                    visible = double_slit["visible"]
                    if visible:
                        sim_state.localised_potential_to_visualize_hartree += tensor
                except KeyError:
                    pass
        except KeyError:
            pass

        try:
            coulomb_potential = sim_state.config["coulomb_potential"]
            print("Creating Coulomb potential.")
            tensor = potential.add_coulomb_potential(
                V=np.zeros(shape=sim_state.tensor_shape, dtype=np.csingle),
                delta_x=sim_state.delta_x_bohr_radii,
                center_bohr_radius=np.array(coulomb_potential["center_bohr_radii_3"]),
                gradient_dir=np.array(coulomb_potential["gradient_direction"]),
                charge_density=coulomb_potential["charge_density_elementary_charge_per_bohr_radius"],
                oxide_start_bohr_radii=coulomb_potential["oxide_start_bohr_radii"],
                oxide_end_bohr_radii=coulomb_potential["oxide_end_bohr_radii"]
            )
            sim_state.localised_potential_hartree += tensor
            visible = coulomb_potential["visible"]
            if visible:
                sim_state.localised_potential_to_visualize_hartree += tensor
        except KeyError:
            print("No Coulomb potential created")

        try:
            gradient = sim_state.config["linear_potential_gradient"]
            print("Creating linear potential gradient.")
            tensor = potential.add_linear_potential_gradient(
                V=np.zeros(shape=sim_state.tensor_shape, dtype=np.csingle),
                delta_x=sim_state.delta_x_bohr_radii,
                center_bohr_radius=np.array(gradient["center_bohr_radii_3"]),
                gradient_dir=np.array(gradient["gradient_direction"]),
                gradient_val=gradient["gradient_magnitude_hartree_per_bohr_radius"],
            )
            sim_state.localised_potential_hartree += tensor
            visible = gradient["visible"]
            if visible:
                sim_state.coulomb_potential = tensor
        except KeyError:
            pass

        np.save(
            file=os.path.join(sim_state.cache_dir, "localized_potential.npy"),
            arr=sim_state.localised_potential_hartree,
        )
        np.save(file=os.path.join(sim_state.cache_dir, "localized_potential_to_visualize.npy"), arr=sim_state.localised_potential_to_visualize_hartree)

    full_init = True
    if sim_state.simulation_method == "fft":
        if use_cache:
            try:
                sim_state.potential_operator = cp.load(file=os.path.join(sim_state.cache_dir, "potential_operator.npy"))
                full_init = False
            except OSError:
                print("No cached potential_operator.npy found.")
        if full_init:
            print("Creating potential operator.")
            sim_state.potential_operator = cp.asarray(operators.init_potential_operator(
                V=sim_state.localised_potential_hartree,
                delta_time=sim_state.delta_time_h_bar_per_hartree,
            ))
            cp.save(file=os.path.join(sim_state.cache_dir, "potential_operator.npy"), arr=sim_state.potential_operator)
    try:
        with open(os.path.join(sim_state.cache_dir, "cached_parameters.toml"), mode="w") as cache_f:
            toml.dump(config, cache_f)
    except OSError as e:
        print("Error while creating parameter cache: " )
        print(e)

    print(
        f"Time spent with initialisation: {(time.time() - initialisation_start_time_s):.2f} s."
    )

    return sim_state
