import numpy as np
import time
import sources.plot as plot
import sources.math_utils as math_utils
import sources.sim_state as sim_st
import sources.measurement as measurement


def time_evolution(wave_tensor, kinetic_operator, potential_operator):
    moment_space_wave_tensor = np.fft.fftn(wave_tensor, norm="forward")
    moment_space_wave_tensor = np.multiply(kinetic_operator, moment_space_wave_tensor)
    wave_tensor = np.fft.fftn(moment_space_wave_tensor, norm="backward")
    wave_tensor = np.multiply(potential_operator, wave_tensor)
    moment_space_wave_tensor = np.fft.fftn(wave_tensor, norm="forward")
    moment_space_wave_tensor = np.multiply(kinetic_operator, moment_space_wave_tensor)
    return np.fft.fftn(moment_space_wave_tensor, norm="backward")


class IterData:
    elapsed_system_time_s = 0.0
    average_iteration_system_time_s = 0.0
    animation_frame_step_interval: int
    png_step_interval: int
    measurement_plane_capture_interval: int
    probability_plot_interval: int
    total_iteration_count: int
    total_simulated_time: float
    per_axis_probability_denisty_plot_interval: int


def run_iteration(sim_state: sim_st.SimState, measurement_tools):
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

    for i in range(iter_data.total_iteration_count + 1):
        iter_start_time_s = time.time()
        print("Iteration: ", i, ".")
        sim_state.probability_density = math_utils.square_of_abs(sim_state.wave_tensor)

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
            sim_state.wave_tensor,
            sim_state.delta_time_h_bar_per_hartree,
        )

        print(
            f"Integral of probability density P = {measurement_tools.measurement_volume_full.get_probability()}."
        )
        print(
            f"Probability of particle in first half P(first) = {measurement_tools.measurement_volume_first_half.get_probability():.3}."
        )
        print(
            f"Probability of particle in second half P(second) = {measurement_tools.measurement_volume_second_half.get_probability():.3}."
        )

        if (
            i % iter_data.per_axis_probability_denisty_plot_interval == 0
            or i % iter_data.animation_frame_step_interval == 0
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
        if i % iter_data.per_axis_probability_denisty_plot_interval == 0:
            measurement_tools.per_axis_density_plot = plot.plot_per_axis_probability_density(
                [
                    measurement_tools.x_axis_probability_density.get_probability_density_with_label(),
                    measurement_tools.y_axis_probability_density.get_probability_density_with_label(),
                    measurement_tools.z_axis_probability_density.get_probability_density_with_label(),
                ],
                delta_x=sim_state.delta_x_bohr_radii,
                delta_t=sim_state.delta_time_h_bar_per_hartree,
                index=i,
                show_fig=False,
            )
        if (
            i % iter_data.animation_frame_step_interval == 0
            or i % iter_data.png_step_interval == 0
        ):
            measurement_tools.canvas.update(
                sim_state.get_view_into_probability_density(),
                iter_count=i,
                delta_time_h_bar_per_hartree=sim_state.delta_time_h_bar_per_hartree,
            )
        if i % iter_data.animation_frame_step_interval == 0:
            measurement_tools.animation_writer_3D.add_frame(
                measurement_tools.canvas.render()
            )
            measurement_tools.animation_writer_per_axis.add_frame(
                measurement_tools.per_axis_density_plot
            )
        if i % iter_data.png_step_interval == 0:
            measurement_tools.canvas.save_to_png(
                f"output/probability_density_{i:04d}.png"
            )
        if i % iter_data.probability_plot_interval == 0:
            plot.plot_probability_evolution(
                [
                    measurement_tools.measurement_volume_full.get_probability_evolution(),
                    measurement_tools.measurement_volume_first_half.get_probability_evolution(),
                    measurement_tools.measurement_volume_second_half.get_probability_evolution(),
                ],
                delta_t=sim_state.delta_time_h_bar_per_hartree,
                index=i,
                show_fig=False,
            )
        if i % iter_data.measurement_plane_capture_interval == 0:
            plot.plot_canvas(
                plane_probability_density=measurement_tools.measurement_plane.get_probability_density(),
                plane_dwell_time_density=measurement_tools.measurement_plane.get_dwell_time(),
                index=i,
            )

        sim_state.wave_tensor = time_evolution(
            wave_tensor=sim_state.wave_tensor,
            kinetic_operator=sim_state.kinetic_operator,
            potential_operator=sim_state.potential_operator,
        )
        iter_time = time.time() - iter_start_time_s
        iter_data.elapsed_system_time_s += iter_time
        print(f"Iteration time: {iter_time} s")

    iter_data.total_simulated_time = (
        sim_state.delta_time_h_bar_per_hartree * iter_data.total_iteration_count
    )
    iter_data.average_iteration_system_time_s = iter_data.elapsed_system_time_s / float(
        iter_data.total_iteration_count
    )
    return sim_state, measurement_tools, iter_data
