import os
import io
import numpy as np
from sources.sim_state import SimState
from sources.iter_data import IterData


def write_snapshot(sim_state: SimState, iter_data: IterData):
    print("Creating snapshot.")
    np.save(arr=sim_state.wave_tensor, file="cache/wave_snapshot.npy")
    with io.open("cache/data_snapshot.txt", mode="w") as f:
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
        )

    sim_state


def read_snapshot(sim_state: SimState, iter_data: IterData):
    try:
        sim_state.wave_tenso = np.load(file="cache/wave_snapshot.npy")
    except OSError:
        print("Failed to read wave tensor!")
    with io.open("cache/data_snapshot.txt", mode="r") as f:
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

    return sim_state, iter_data


def remove_snapshot():
    os.remove("cache/data_snapshot.txt")
    os.remove("cache/wave_snapshot.npy")
