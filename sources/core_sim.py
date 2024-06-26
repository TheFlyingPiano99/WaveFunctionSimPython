import numpy as np
import cupy as cp
import time
import sources.plot as plot
import sources.math_utils as math_utils
from sources.sim_state import SimulationMethod, SimState
from sources.measurement import MeasurementTools
from alive_progress import alive_bar
import sources.signal_handling as signal_handling
import os
from sources.iter_data import IterData
import keyboard
import sys
from colorama import Fore, Style
from tqdm import tqdm
import sources.snapshot_io as snapshot_io


def time_step(sim_state: SimState,
              measurement_tools: MeasurementTools,
              iter_data: IterData):
    measurement_tools.measure_and_render(sim_state, iter_data)
    sim_state.evolve_state()
    sim_state.update_potential()

def run_iteration(sim_state: SimState, measurement_tools: MeasurementTools, iter_data: IterData):
    print(Fore.GREEN + "Simulating " + Style.RESET_ALL + "(Press <Ctrl-c> to quit.)")

    iter_data.is_quit = False
    signal_handling.register_signal_handler(iter_data)

    start_index = iter_data.i  # Needed because of snapshots

    # Main iteration loop:
    """
    # This progress bar was problematic in some consoles:
    with alive_bar(iter_data.total_iteration_count) as bar:
        for j in range(iter_data.i):
            bar()
    """
    with tqdm(total=iter_data.total_iteration_count) as progress_bar:
        for iter_data.i in range(start_index, iter_data.total_iteration_count):
            iter_start_time_s = time.time()

            # Do simulation step:
            time_step(sim_state, measurement_tools, iter_data)

            # update time variable and progress bar:
            iter_data.elapsed_system_time_s += time.time() - iter_start_time_s
            progress_bar.n = iter_data.i
            progress_bar.refresh()
            if iter_data.is_quit:   # Quit simulation
                snapshot_io.write_snapshot(sim_state, iter_data)
                measurement_tools.finish(sim_state)
                sys.exit(0)

    # Calculate resulting time statistics after the iteration:
    iter_data.total_simulated_time = (
        sim_state.get_delta_time_h_bar_per_hartree() * iter_data.total_iteration_count
    )
    iter_data.average_iteration_system_time_s = iter_data.elapsed_system_time_s / float(
        iter_data.total_iteration_count
    )

    return sim_state, measurement_tools, iter_data
