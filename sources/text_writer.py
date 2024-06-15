from sources.sim_state import SimState
import sources.core_sim as core_sim
import io
from datetime import timedelta
from colorama import Fore, Back, Style
import os


def get_title_text():
    text = io.StringIO()
    text.write(Fore.GREEN + "\n ____________________________\n")
    text.write("|                            |\n")
    text.write("|   Wave packet simulation   |\n")
    text.write("|____________________________|\n" + Style.RESET_ALL)
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

def write_sim_state(sim_state: SimState):
    with open(os.path.join(sim_state.get_output_dir(), "parameters.txt"), mode="w") as f:
        f.write(sim_state.get_sim_state_description_text(sim_state))
        f.write(sim_state.get_potential_description_text(sim_state))
        f.write(sim_state.get_simulation_method_text(sim_state))


def append_iter_data(iter_data: core_sim.IterData, sim_state: SimState):
    with open(os.path.join(sim_state.get_output_dir(), "parameters.txt"), mode="a") as f:
        f.write("\n" + get_finish_text(iter_data))

def get_help_text():
    text = io.StringIO()
    text.write("This is a simulation software for wave packet simulation created by Zoltan Simon.\n\n\n")
    text.write("Available flags and parameters:\n\n")
    text.write("-h \t\t\t... See help\n")
    text.write("--help \t\t\t... See help\n")
    text.write("-nc \t\t\t... Run in No Cache mode\n")
    text.write("--version \t\t... See current version\n")
    return text.getvalue()
def get_version_text():
    return "beta version\n"
