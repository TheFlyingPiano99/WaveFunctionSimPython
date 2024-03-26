import os
import io
import numpy as np
import cupy as cp
from sources.sim_state import SimState
from sources.iter_data import IterData


def write_snapshot(sim_state: SimState, iter_data: IterData):
    print("\nCreating snapshot. Please wait for the application to exit by itself!")
    try:
        cp.save(arr=sim_state.wave_tensor, file=os.path.join(sim_state.cache_dir, "wave_snapshot.npy"))
    except IOError:
        print("\nFailed writing snapshot of the wave function.")

    with io.open(os.path.join(sim_state.cache_dir, "data_snapshot.txt"), mode="w") as f:
        f.write(
            f"{iter_data.i};"
            f"{iter_data.elapsed_system_time_s};"
            f"{iter_data.average_iteration_system_time_s};"
            f"{iter_data.animation_frame_step_interval};"
            f"{iter_data.png_step_interval};"
            f"{iter_data.measurement_plane_capture_interval};"
            f"{iter_data.probability_plot_interval};"
            f"{iter_data.total_iteration_count};"
            f"{iter_data.total_simulated_time};"
            f"{iter_data.per_axis_probability_denisty_plot_interval};"
            f"{iter_data.wave_function_save_interval};"
        )

    sim_state


def read_snapshot(sim_state: SimState, iter_data: IterData):
    if os.path.exists(os.path.join(sim_state.cache_dir, "wave_snapshot.npy")):
        try:
            wave_snapshot = np.load(file=os.path.join(sim_state.cache_dir, "wave_snapshot.npy"))
            sim_state.wave_tensor = cp.asarray(wave_snapshot)
        except:
            print("No usable previous wave tensor found!")
    try:
        with io.open(os.path.join(sim_state.cache_dir, "data_snapshot.txt"), mode="r") as f:
            data = f.read()
            split_data = data.split(";")
            iter_data.i = int(split_data[0])
            iter_data.elapsed_system_time_s = float(split_data[1])
            iter_data.average_iteration_system_time_s = float(split_data[2])
            iter_data.animation_frame_step_interval = int(split_data[3])
            iter_data.png_step_interval = int(split_data[4])
            iter_data.measurement_plane_capture_interval = int(split_data[5])
            iter_data.probability_plot_interval = int(split_data[6])
            iter_data.total_iteration_count = int(split_data[7])
            iter_data.total_simulated_time = float(split_data[8])
            iter_data.per_axis_probability_denisty_plot_interval = int(split_data[9])
            iter_data.wave_function_save_interval = int(split_data[10])
    except IOError:
        print("Failed to read " + os.path.join(sim_state.cache_dir, "data_snapshot.txt"))
    return sim_state, iter_data


def remove_snapshot(sim_state: SimState):
    if os.path.exists(os.path.join(sim_state.cache_dir, "data_snapshot.txt")):
        os.remove(os.path.join(sim_state.cache_dir, "data_snapshot.txt"))
    if os.path.exists(os.path.join(sim_state.cache_dir, "wave_snapshot.npy")):
        os.remove(os.path.join(sim_state.cache_dir, "wave_snapshot.npy"))
