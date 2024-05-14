import numpy as np
import cupy as cp
import time
import sources.plot as plot
import sources.math_utils as math_utils
import sources.sim_state as sim_st
import sources.measurement as measurement
from alive_progress import alive_bar
import sources.signal_handling as signal_handling
import os
import sources.snapshot_io as snapshot_io
from sources.iter_data import IterData
import keyboard
import sys
from colorama import Fore, Style
from tqdm import tqdm


def fft_time_evolution(wave_tensor, kinetic_operator, potential_operator):
    moment_space_wave_tensor = cp.fft.fftn(wave_tensor, norm="forward")
    moment_space_wave_tensor = cp.multiply(kinetic_operator, moment_space_wave_tensor)
    wave_tensor = cp.fft.fftn(moment_space_wave_tensor, norm="backward")
    wave_tensor = cp.multiply(potential_operator, wave_tensor)
    moment_space_wave_tensor = cp.fft.fftn(wave_tensor, norm="forward")
    moment_space_wave_tensor = cp.multiply(kinetic_operator, moment_space_wave_tensor)
    return cp.fft.fftn(moment_space_wave_tensor, norm="backward")


def merged_fft_time_evolution(
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


def power_series_time_evolution(sim_state: sim_st.SimState, p: int, next_s_kernel: cp.ElementwiseKernel, s, v: cp.ndarray):
    shape = sim_state.wave_tensor.shape
    grid_size = (64, 64, 64)
    block_size = (shape[0] // grid_size[0], shape[1] // grid_size[1], shape[2] // grid_size[2])
    pingpong_idx = 0
    s[1 - pingpong_idx] = cp.copy(sim_state.wave_tensor)
    for n in range(1, p + 1):
        next_s_kernel(
            grid_size,
            block_size,
            (
                s[1 - pingpong_idx],
                s[pingpong_idx],
                v,
                sim_state.wave_tensor,
                cp.double(sim_state.delta_time_h_bar_per_hartree) if sim_state.double_precision_wave_tensor
                else cp.float32(sim_state.delta_time_h_bar_per_hartree),
                cp.double(sim_state.delta_x_bohr_radii) if sim_state.double_precision_wave_tensor
                else cp.float32(sim_state.delta_x_bohr_radii),
                cp.double(sim_state.particle_mass) if sim_state.double_precision_wave_tensor
                else cp.float32(sim_state.particle_mass),
                cp.int32(shape[0]),
                cp.int32(n)
            )
        )
        pingpong_idx = 1 - pingpong_idx
    return sim_state.wave_tensor


def init_iter_data(sim_state: sim_st.SimState):
    iter_data = IterData()
    iter_data.animation_frame_step_interval = sim_state.config["iteration"]["animation_frame_step_interval"]
    iter_data.png_step_interval = sim_state.config["iteration"]["png_step_interval"]
    iter_data.measurement_plane_capture_interval = sim_state.config["iteration"]["measurement_plane_capture_interval"]
    iter_data.probability_plot_interval = sim_state.config["iteration"]["probability_plot_interval"]
    iter_data.total_iteration_count = sim_state.config["iteration"]["total_iteration_count"]
    iter_data.per_axis_probability_denisty_plot_interval = sim_state.config["iteration"]["per_axis_probability_denisty_interval"]
    iter_data.wave_function_save_interval = sim_state.config["iteration"]["wave_function_save_interval"]
    return iter_data

def measure_and_render(iter_data, sim_state: sim_st.SimState, measurement_tools):
    # Update all measurement tools:
    measurement_tools.measurement_volume_full.integrate_probability_density(
        sim_state.probability_density
    )
    '''
    measurement_tools.measurement_volume_first_half.integrate_probability_density(
        sim_state.probability_density
    )
    measurement_tools.measurement_volume_second_half.integrate_probability_density(
        sim_state.probability_density
    )
    '''

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

    # Plot state:
    if iter_data.i % iter_data.per_axis_probability_denisty_plot_interval == 0:
        measurement_tools.per_axis_density_plot = plot.plot_per_axis_probability_density(
            out_dir=sim_state.output_dir,
            title=sim_state.config["view"]["per_axis_plot"]["title"],
            data=[
                measurement_tools.x_axis_probability_density.get_probability_density_with_label(),
                measurement_tools.y_axis_probability_density.get_probability_density_with_label(),
                measurement_tools.z_axis_probability_density.get_probability_density_with_label(),
                measurement_tools.projected_probability.get_probability_density_with_label(),
            ],
            delta_x=sim_state.delta_x_bohr_radii,
            delta_t=sim_state.delta_time_h_bar_per_hartree,
            potential_scale=sim_state.config["view"]["per_axis_plot"]["potential_plot_scale"],
            index=iter_data.i,
            show_fig=False,
        )
    if (
            iter_data.i % iter_data.animation_frame_step_interval == 0
            or iter_data.i % iter_data.png_step_interval == 0
    ):
        measurement_tools.volumetric.update(
            sim_state.get_view_into_probability_density(),
            iter_count=iter_data.i,
            delta_time_h_bar_per_hartree=sim_state.delta_time_h_bar_per_hartree,
        )
    if iter_data.i % iter_data.animation_frame_step_interval == 0:
        measurement_tools.animation_writer_3D.add_frame(
            measurement_tools.volumetric.render()
        )
        measurement_tools.animation_writer_per_axis.add_frame(
            measurement_tools.per_axis_density_plot
        )
    if iter_data.i % iter_data.png_step_interval == 0:
        measurement_tools.volumetric.render_to_png(out_dir=sim_state.output_dir, index=iter_data.i)
    if iter_data.i % iter_data.probability_plot_interval == 0:
        plot.plot_probability_evolution(
            out_dir=sim_state.output_dir,
            probability_evolutions=[
                measurement_tools.measurement_volume_full.get_probability_evolution(),
                #measurement_tools.measurement_volume_first_half.get_probability_evolution(),
                #measurement_tools.measurement_volume_second_half.get_probability_evolution(),
            ],
            delta_t=sim_state.delta_time_h_bar_per_hartree,
            index=iter_data.i,
            show_fig=False,
        )
    if iter_data.i % iter_data.measurement_plane_capture_interval == 0:
        plot.plot_canvas(
            out_dir=sim_state.output_dir,
            plane_probability_density=measurement_tools.measurement_plane.get_probability_density(),
            plane_dwell_time_density=measurement_tools.measurement_plane.get_dwell_time(),
            index=iter_data.i,
            delta_x=sim_state.delta_x_bohr_radii,
            delta_t=sim_state.delta_time_h_bar_per_hartree
        )


def write_wave_function_to_file(sim_state: sim_st.SimState, iter_data):
    if iter_data.i % iter_data.wave_function_save_interval == 0:
        if not os.path.exists(os.path.join(sim_state.output_dir, f"wave_function")):
            os.makedirs(os.path.join(sim_state.output_dir, f"wave_function"), exist_ok=True)
        try:
            cp.save(arr=sim_state.get_view_into_raw_wave_function(), file=os.path.join(sim_state.output_dir, f"wave_function/wave_function_{iter_data.i:04d}.npy"))
        except IOError:
            print(Fore.RED + "\nERROR: Failed writing file: "+ os.path.join(sim_state.output_dir, f"wave_function/wave_function_{iter_data.i:04d}.npy") + Style.RESET_ALL)


def run_iteration(sim_state: sim_st.SimState, measurement_tools):
    probability_evolutions = []
    probability_divergence_iteration_count = []
    is_diverged = False
    copy_of_initial_wave_function = cp.copy(sim_state.wave_tensor)
    min_p = 0 if sim_state.simulation_method == "fft" else 10
    max_p = 0 if sim_state.simulation_method == "fft" else 10
    divergence_idx_file = open(os.path.join(sim_state.output_dir, "iteration_idx_of_divergence.txt"), 'w')
    for p in range(min_p, max_p + 1):  # For the p iteration
        is_diverged = False # For comparison
        sim_state.simulation_method = "fft" if p == 0 else "power_series" # For p comparison
        sim_state.wave_tensor = cp.copy(copy_of_initial_wave_function)  # For p comparison
        if sim_state.enable_visual_output:
            measurement_tools.measurement_volume_full.clear()   # For p comparison
        # Setup iteration parameters:
        iter_data = init_iter_data(sim_state)
        if (
            sim_state.use_cache
            and os.path.exists(os.path.join(sim_state.cache_dir, "data_snapshot.txt"))
            and os.path.exists(os.path.join(sim_state.cache_dir, "wave_snapshot.npy"))
        ):
            sim_state, iter_data = snapshot_io.read_snapshot(sim_state, iter_data)
            snapshot_io.remove_snapshot(sim_state)
            print(Fore.BLUE + "Snapshot of an interrupted simulation loaded.\nResuming previous wave function.\n" + Style.RESET_ALL)

        print(Fore.GREEN + "Simulating " + Style.RESET_ALL + "(Press <Ctrl-c> to quit.)")

        iter_data.is_quit = False
        signal_handling.register_signal_handler(iter_data)

        start_index = iter_data.i  # Needed because of snapshots

        if sim_state.simulation_method == "power_series":   # Define the kernel
            format_kernel_source = r'''
                #include <cupy/complex.cuh>
                
                extern "C" __global__
                void next_s(
                    complex<{0}>* s_prev, complex<{0}>* s_next, const complex<float>* v, complex<{0}>* wave_function,
                    {0} delta_t, {0} delta_x, {0} mass, int array_size, int n
                )
                {{
                    int k = blockIdx.x * blockDim.x + threadIdx.x;
                    int j = blockIdx.y * blockDim.y + threadIdx.y;
                    int i = blockIdx.z * blockDim.z + threadIdx.z;
                    int idx = i * array_size * array_size + j * array_size + k;
                    int idx_n00 = 0;
                    if (i > 0) {{
                        idx_n00 = (i - 1) * array_size * array_size + j * array_size + k; 
                    }}
                    else {{
                        idx_n00 = (i) * array_size * array_size + j * array_size + k; 
                    }}
                    int idx_p00 = 0;
                    if (i < array_size - 1) {{
                        idx_p00 = (i + 1) * array_size * array_size + j * array_size + k; 
                    }}
                    else {{
                        idx_p00 = (i) * array_size * array_size + j * array_size + k; 
                    }}
                    int idx_0n0 = 0;
                    if (j > 0) {{
                        idx_0n0 = i * array_size * array_size + (j - 1) * array_size + k; 
                    }}
                    else {{
                        idx_0n0 = i * array_size * array_size + (j) * array_size + k; 
                    }}
                    int idx_0p0 = 0;
                    if (j < array_size - 1) {{
                        idx_0p0 = i * array_size * array_size + (j + 1) * array_size + k; 
                    }}
                    else {{
                        idx_0p0 = i * array_size * array_size + (j) * array_size + k; 
                    }}
                    int idx_00n = 0;
                    if (k > 0) {{
                        idx_00n = i * array_size * array_size + j * array_size + k - 1; 
                    }}
                    else {{
                        idx_00n = i * array_size * array_size + j * array_size + k; 
                    }}
                    int idx_00p = 0;
                    if (k < array_size - 1) {{
                        idx_00p = i * array_size * array_size + j * array_size + k + 1; 
                    }}
                    else {{
                        idx_00p = i * array_size * array_size + j * array_size + k; 
                    }}
    
                    complex<{0}> laplace_s = (
                          s_prev[idx_n00]
                        + s_prev[idx_p00]
                        + s_prev[idx_0n0]
                        -6.0{1} * s_prev[idx]
                        + s_prev[idx_0p0]
                        + s_prev[idx_00n]
                        + s_prev[idx_00p]
                    ) / delta_x / delta_x;
                    
                    complex<{0}> s = complex<{0}>(0.0{1}, 1.0{1}) * complex<{0}>(delta_t / ({0})n, 0.0{1})
                        * (complex<{0}>(1.0{1} / 2.0{1} / mass, 0.0{1}) * laplace_s - complex<{0}>(v[idx]) * s_prev[idx]);
                    s_next[idx] = s;
                    wave_function[idx] += s; 
                }}
            '''.format("double" if sim_state.double_precision_wave_tensor else "float", "" if sim_state.double_precision_wave_tensor else "f")
            #print("Kernel source:")
            #print(format_kernel_source)
            next_s_kernel = cp.RawKernel(format_kernel_source,
            'next_s',
            enable_cooperative_groups=False)
            s = [
                cp.zeros(shape=sim_state.wave_tensor.shape, dtype=sim_state.wave_tensor.dtype),
                cp.zeros(shape=sim_state.wave_tensor.shape, dtype=sim_state.wave_tensor.dtype)
            ]   # s is used as a pair of pingpong buffers to store power series elements
            v = cp.asarray(sim_state.localised_potential_hartree)   # Copy to GPU

        # Main iteration loop:
        """
        # This progress bar was problematic in some consoles:
        with alive_bar(iter_data.total_iteration_count) as bar:
            for j in range(iter_data.i):
                bar()
        """
        with tqdm(total=iter_data.total_iteration_count) as progress_bar:
            for iter_data.i in range(start_index, iter_data.total_iteration_count):
                if iter_data.is_quit:
                    snapshot_io.write_snapshot(sim_state, iter_data)
                    sys.exit(0)

                iter_start_time_s = time.time()
                sim_state.probability_density = math_utils.square_of_abs(
                    sim_state.wave_tensor
                )

                if sim_state.enable_wave_function_save:
                    write_wave_function_to_file(sim_state=sim_state, iter_data=iter_data)

                if sim_state.enable_visual_output:
                    measure_and_render(iter_data, sim_state, measurement_tools)

                # Main time development step:
                if sim_state.simulation_method == "fft":
                    sim_state.wave_tensor = fft_time_evolution(
                        wave_tensor=sim_state.wave_tensor,
                        kinetic_operator=sim_state.kinetic_operator,
                        potential_operator=sim_state.potential_operator,
                    )
                elif sim_state.simulation_method == "power_series":
                    power_series_time_evolution(sim_state=sim_state, p=p, next_s_kernel=next_s_kernel, s=s, v=v)
                else:
                    print("ERROR: Undefined simulation method")

                iter_time = time.time() - iter_start_time_s
                iter_data.elapsed_system_time_s += iter_time
                progress_bar.n = iter_data.i
                progress_bar.refresh()

                # For comparison:
                if min_p != max_p:
                    if sim_state.enable_visual_output:
                        measurement_tools.measurement_volume_full.integrate_probability_density(
                            sim_state.probability_density
                        )
                    probability = cp.sum(sim_state.probability_density)
                    if not is_diverged and probability > 2.0:
                        is_diverged = True
                        probability_divergence_iteration_count.append(iter_data.i)
                    if not is_diverged:
                        probability_divergence_iteration_count.append("convergent")

        # Calculate resulting time statistics after the iteration:
        iter_data.total_simulated_time = (
            sim_state.delta_time_h_bar_per_hartree * iter_data.total_iteration_count
        )
        iter_data.average_iteration_system_time_s = iter_data.elapsed_system_time_s / float(
            iter_data.total_iteration_count
        )

        if sim_state.enable_visual_output:
            evolution = list(measurement_tools.measurement_volume_full.get_probability_evolution())
            evolution[1] = f"p = {p}" if sim_state.simulation_method == "power_series" else "SOF"
            probability_evolutions.append(evolution)
            plot.plot_probability_evolution(
                out_dir=os.path.join(sim_state.output_dir, "probability_comparison"),
                probability_evolutions=probability_evolutions,
                index=p,
                delta_t=sim_state.delta_time_h_bar_per_hartree
            )
            for_frame = {}
            time_steps = []
            for item in probability_evolutions:
                for_frame[item[1]] = item[0].tolist()

            for i in range(0, iter_data.total_iteration_count):
                time_steps.append(i * sim_state.delta_time_h_bar_per_hartree)

        divergence_idx_file.write(str(probability_divergence_iteration_count[-1]) + "\n")
    divergence_idx_file.close()
    return sim_state, measurement_tools, iter_data
