import sources.sim_state as sim_st
import sources.core_sim as core_sim


def write_sim_state(sim_state: sim_st.SimState):
    with open("output/parameters.txt", mode="w") as f:
        f.write("Simulation parameters\n")
        f.write("\n")
        f.write(
            "___________________________________________________________________________\n"
        )
        f.write("\n")
        f.write("Wave packet:\n")
        f.write(f"Particle mass: {sim_state.particle_mass} electron rest mass\n")
        f.write(
            f"Initial mean position: {sim_state.initial_wp_position_bohr_radii_3} Bohr radii\n"
        )
        f.write(
            f"Initial mean momentum: {sim_state.initial_wp_momentum_h_per_bohr_radius} h-bar / Bohr radius\n"
        )
        f.write(
            "___________________________________________________________________________\n"
        )
        f.write("\n")
        f.write("Volume:\n")
        f.write(
            f"Width of simulated volume: {sim_state.simulated_volume_width_bohr_radii:.4f} Bohr radii\n"
        )
        f.write(f"Number of samples per axis: {sim_state.N}\n")
        f.write(f"Grid step size: {sim_state.delta_x_bohr_radii:.4f} Bohr radii\n")
        f.write(
            "___________________________________________________________________________\n"
        )
        f.write("\n")
        f.write("Iteration:\n")
        f.write(
            f"Delta time: {sim_state.delta_time_h_bar_per_hartree:.4f} h-bar / hartree\n"
        )


def append_iter_data(iter_data: core_sim.IterData):
    with open("output/parameters.txt", mode="a") as f:
        f.write(f"Total iteration count: {iter_data.total_iteration_count}\n")
        f.write(
            f"Total simulated time: {iter_data.total_simulated_time:.4f} h-bar / hartree\n"
        )
        f.write(
            f"Elapsed system time during iteration: {iter_data.elapsed_system_time_s:.4f} s\n"
        )
        f.write(
            f"Average system time of an iteration: {iter_data.average_iteration_system_time_s:.4f} s\n"
        )
