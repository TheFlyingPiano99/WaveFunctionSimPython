import time

import toml
from sources.sim_state import SimState
import sources.text_writer as text_writer
import os
from colorama import Fore, Style
import sys
from sources.measurement import MeasurementTools

def is_toml(file_name):
    return file_name.endswith('.toml')

def initialize(use_cache: bool = True):
    initialisation_start_time_s = time.time()
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
            sim_state = SimState(config)
            try:
                if not os.path.exists(sim_state.get_cache_dir()):
                    os.makedirs(sim_state.get_cache_dir(), exist_ok=True)
                with open(os.path.join(sim_state.get_cache_dir(), "cached_parameters.toml"), mode="r") as cache_f:
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

    sim_state.set_use_cache(use_cache)


    # Warn user about not empty output directory:
    if not os.path.exists(sim_state.get_output_dir()):
        os.makedirs(sim_state.get_output_dir(), exist_ok=True)
    if os.listdir(sim_state.get_output_dir()):
        answer = ""
        while not answer in {"y", "n"}:
            print("\n" + Fore.RED + "Output directory is not empty!\n"
                'Continuing will possibly override files under "' + sim_state.get_output_dir() + '".' + Style.RESET_ALL + "\n"
                                                                              "Would you still like to continue [y/n]?",
                end=" ",
                )
            answer = input()
            if answer == "n":
                print("Exiting application.")
                sys.exit(0)

    print(
        "\n***************************************************************************************\n"
    )

    sim_state.initialize_state()

    measurement_tools = MeasurementTools(config, sim_state)

    try:
        with open(os.path.join(sim_state.get_cache_dir(), "cached_parameters.toml"), mode="w") as cache_f:
            toml.dump(config, cache_f)
    except OSError as e:
        print("Error while creating parameter cache: ")
        print(e)

    print(
        f"Time spent with initialisation: {(time.time() - initialisation_start_time_s):.2f} s.\n"
        # Extra new line at the end.
    )

    print(sim_state.get_simulation_method_text(use_colors=True))


    return sim_state, measurement_tools
