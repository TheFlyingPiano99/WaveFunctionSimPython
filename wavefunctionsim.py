import sources.initialisation
from sources import volume_visualization, animation, measurement, text_writer
import sources.initialisation as init
import sources.core_sim as core_sim


class MeasurementTools:
    elapsed_iter_time = 0.0
    measurement_plane: measurement.MeasurementPlane


def sim():
    print("Wave function simulation")

    sim_state = init.initialize()

    print(
        "***************************************************************************************"
    )
    print("Starting simulation")
    measurement_tools = MeasurementTools()
    measurement_tools.canvas = volume_visualization.VolumeCanvas(
        volume_data=sim_state.probability_density,
        secondary_data=sim_state.only_the_obstacle_potential,
    )
    measurement_tools.animation_writer = animation.AnimationWriter(
        "output/probability_density_time_development.gif"
    )
    measurement_tools.measurement_plane = measurement.MeasurementPlane(
        wave_tensor=sim_state.wave_tensor,
        delta_x=sim_state.delta_x_bohr_radii,
        location_bohr_radii=25.0,
    )
    measurement_tools.measurement_volume_full = measurement.AAMeasurementVolume(
        bottom_corner=(0, 0, 0),
        top_corner=(sim_state.N, sim_state.N, sim_state.N),
        label="Full volume",
    )
    measurement_tools.measurement_volume_first_half = measurement.AAMeasurementVolume(
        bottom_corner=(0, 0, 0),
        top_corner=(sim_state.N, sim_state.N, int(sim_state.N / 2)),
        label="First half",
    )
    measurement_tools.measurement_volume_second_half = measurement.AAMeasurementVolume(
        bottom_corner=(0, 0, int(sim_state.N / 2)),
        top_corner=(sim_state.N, sim_state.N, sim_state.N),
        label="Second half",
    )

    # Run simulation
    sim_state, measurement_tools, iter_data = core_sim.run_iteration(
        sim_state, measurement_tools
    )

    measurement_tools.animation_writer.finish()
    print("Simulation has finished.")
    print(f"Total simulation time: {iter_data.elapsed_system_time_s} s")
    print(f"Average iteration time: {iter_data.average_iteration_system_time_s} s")
    text_writer.append_iter_data(iter_data)


if __name__ == "__main__":
    sim()
