import numpy as np
import time
import sources.plot as plot
import sources.math_utils as math_utils
import sources.sim_state as sim_st
import sources.measurement as measurement
from alive_progress import alive_bar
import sources.signal_handling as signal_handling
import os
import sources.snapshot_io as snapshot
from sources.iter_data import IterData
import keyboard
import sys
from colorama import Fore, Style
import cupy as cp


def time_evolution(wave_tensor, kinetic_operator, potential_operator):
    moment_space_wave_tensor = cp.fft.fftn(wave_tensor, norm="forward")
    moment_space_wave_tensor = cp.multiply(kinetic_operator, moment_space_wave_tensor)
    wave_tensor = cp.fft.fftn(moment_space_wave_tensor, norm="backward")
    wave_tensor = cp.multiply(potential_operator, wave_tensor)
    moment_space_wave_tensor = cp.fft.fftn(wave_tensor, norm="forward")
    moment_space_wave_tensor = cp.multiply(kinetic_operator, moment_space_wave_tensor)
    return cp.fft.fftn(moment_space_wave_tensor, norm="backward")


def merged_time_evolution(
    wave_tensor,
    kinetic_operator,
    potential_operator,
    merged_kinetic_operator,
    merged_iteration_count,
):
    moment_space_wave_tensor = cp.fft.fftn(wave_tensor, norm="forward")
    moment_space_wave_tensor = cp.multiply(kinetic_operator, moment_space_wave_tensor)

    for i in range(merged_iteration_count - 1):
        wave_tensor = cp.fft.fftn(moment_space_wave_tensor, norm="backward")
        wave_tensor = cp.multiply(potential_operator, wave_tensor)
        moment_space_wave_tensor = cp.fft.fftn(wave_tensor, norm="forward")
        moment_space_wave_tensor = cp.multiply(
            merged_kinetic_operator, moment_space_wave_tensor
        )

    wave_tensor = cp.fft.fftn(moment_space_wave_tensor, norm="backward")
    wave_tensor = cp.multiply(potential_operator, wave_tensor)
    moment_space_wave_tensor = cp.fft.fftn(wave_tensor, norm="forward")
    moment_space_wave_tensor = cp.multiply(kinetic_operator, moment_space_wave_tensor)
    return cp.fft.fftn(moment_space_wave_tensor, norm="backward")


def run_iteration(sim_state: sim_st.SimState, measurement_tools):
    # Setup iteration parameters:
    iter_data = IterData()
    iter_data.animation_frame_step_interval = sim_state.config["Iteration"][
        "animation_frame_step_interval"
    ]
    iter_data.png_step_interval = sim_state.config["Iteration"]["png_step_interval"]
    iter_data.measurement_plane_capture_interval = sim_state.config["Iteration"][
        "measurement_plane_capture_interval"
    ]
    iter_data.probability_plot_interval = sim_state.config["Iteration"][
        "probability_plot_interval"
    ]
    iter_data.total_iteration_count = sim_state.config["Iteration"][
        "total_iteration_count"
    ]
    iter_data.per_axis_probability_denisty_plot_interval = sim_state.config[
        "Iteration"
    ]["per_axis_probability_denisty_interval"]

    if (
        sim_state.use_cache
        and os.path.exists("cache/data_snapshot.txt")
        and os.path.exists("cache/wave_snapshot.npy")
    ):
        print("Snapshot of interrupted simulation detected.")
        sim_state, iter_data = snapshot.read_snapshot(sim_state, iter_data)
        snapshot.remove_snapshot()

    print(Fore.GREEN + "Simulating " + Style.RESET_ALL + "(Press <Ctrl-c> to quit.)")

    iter_data.is_quit = False
    signal_handling.register_signal_handler(iter_data)

    start_index = iter_data.i  # Needed because of snapshots

    # Main iteration loop:
    with alive_bar(iter_data.total_iteration_count) as bar:
        for j in range(iter_data.i):
            bar()
        for iter_data.i in range(start_index, iter_data.total_iteration_count):
            if iter_data.is_quit:
                snapshot.write_snapshot(sim_state, iter_data)
                sys.exit(0)

            iter_start_time_s = time.time()
            sim_state.probability_density = math_utils.square_of_abs(
                sim_state.wave_tensor
            )

            measurement_tools.measurement_volume_full.integrate_probability_density(
                sim_state.probability_density
            )
            measurement_tools.measurement_volume_first_half.integrate_probability_density(
                sim_state.probability_density
            )
            measurement_tools.measurement_volume_second_half.integrate_probability_density(
                sim_state.probability_density
            )

            measurement_tools.measurement_plane.integrate(
                sim_state.probability_density,
                sim_state.delta_time_h_bar_per_hartree,
            )

            if (
                iter_data.i % iter_data.per_axis_probability_denisty_plot_interval == 0
                or iter_data.i % iter_data.animation_frame_step_interval == 0
            ):
                measurement_tools.x_axis_probability_density.integrate_probability_density(
                    sim_state.probability_density
                )
                measurement_tools.y_axis_probability_density.integrate_probability_density(
                    sim_state.probability_density
                )
                measurement_tools.z_axis_probability_density.integrate_probability_density(
                    sim_state.probability_density
                )
            if iter_data.i % iter_data.per_axis_probability_denisty_plot_interval == 0:
                measurement_tools.per_axis_density_plot = plot.plot_per_axis_probability_density(
                    [
                        measurement_tools.x_axis_probability_density.get_probability_density_with_label(),
                        measurement_tools.y_axis_probability_density.get_probability_density_with_label(),
                        measurement_tools.z_axis_probability_density.get_probability_density_with_label(),
                    ],
                    delta_x=sim_state.delta_x_bohr_radii,
                    delta_t=sim_state.delta_time_h_bar_per_hartree,
                    index=iter_data.i,
                    show_fig=False,
                )
            if (
                iter_data.i % iter_data.animation_frame_step_interval == 0
                or iter_data.i % iter_data.png_step_interval == 0
            ):
                measurement_tools.canvas.update(
                    sim_state.get_view_into_probability_density(),
                    iter_count=iter_data.i,
                    delta_time_h_bar_per_hartree=sim_state.delta_time_h_bar_per_hartree,
                )
            if iter_data.i % iter_data.animation_frame_step_interval == 0:
                measurement_tools.animation_writer_3D.add_frame(
                    measurement_tools.canvas.render()
                )
                measurement_tools.animation_writer_per_axis.add_frame(
                    measurement_tools.per_axis_density_plot
                )
            if iter_data.i % iter_data.png_step_interval == 0:
                measurement_tools.canvas.render_to_png(iter_data.i)
            if iter_data.i % iter_data.probability_plot_interval == 0:
                plot.plot_probability_evolution(
                    [
                        measurement_tools.measurement_volume_full.get_probability_evolution(),
                        measurement_tools.measurement_volume_first_half.get_probability_evolution(),
                        measurement_tools.measurement_volume_second_half.get_probability_evolution(),
                    ],
                    delta_t=sim_state.delta_time_h_bar_per_hartree,
                    index=iter_data.i,
                    show_fig=False,
                )
            if iter_data.i % iter_data.measurement_plane_capture_interval == 0:
                plot.plot_canvas(
                    plane_probability_density=measurement_tools.measurement_plane.get_probability_density(),
                    plane_dwell_time_density=measurement_tools.measurement_plane.get_dwell_time(),
                    index=iter_data.i,
                )

            sim_state.wave_tensor = time_evolution(
                wave_tensor=sim_state.wave_tensor,
                kinetic_operator=sim_state.kinetic_operator,
                potential_operator=sim_state.potential_operator,
            )
            iter_time = time.time() - iter_start_time_s
            iter_data.elapsed_system_time_s += iter_time
            bar()

    # Calculate resulting time statistics:
    iter_data.total_simulated_time = (
        sim_state.delta_time_h_bar_per_hartree * iter_data.total_iteration_count
    )
    iter_data.average_iteration_system_time_s = iter_data.elapsed_system_time_s / float(
        iter_data.total_iteration_count
    )
    return sim_state, measurement_tools, iter_data
