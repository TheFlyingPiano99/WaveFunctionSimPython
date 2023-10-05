from sources import volume_visualization, animation, measurement, text_writer
import sources.initialisation as init
import sources.core_sim as core_sim
import sys
import os
import signal


class MeasurementTools:
    measurement_plane: measurement.MeasurementPlane
    canvas: volume_visualization.VolumeCanvas
    animation_writer_3D: animation.AnimationWriter
    animation_writer_per_axis: animation.AnimationWriter
    x_axis_probability_density: measurement.ProjectedMeasurement
    y_axis_probability_density: measurement.ProjectedMeasurement
    z_axis_probability_density: measurement.ProjectedMeasurement


def sim():
    print("Wave packet simulation")
    print("Created by Zoltan Simon")

    out_dir = "output/"
    if os.path.exists(out_dir):
        if os.listdir(out_dir):
            answer = ""
            while not answer in {"y", "n"}:
                print(
                    "Output directory is not empty.\n"
                    "os.listdir('/your/path') will override previous files under \"./output/\".\n"
                    "Would you like to still continue [y/n]?",
                    end=" ",
                )
                answer = input()
                if answer == "n":
                    print("Exiting application.")
                    sys.exit(0)

    sim_state = init.initialize()
    text_writer.write_sim_state(sim_state)

    print(
        "****************************************************************************"
    )
    print("Simulating (Press <Ctrl-c> to quit.)")
    measurement_tools = MeasurementTools()
    measurement_tools.canvas = volume_visualization.VolumeCanvas(
        volume_data=sim_state.get_view_into_probability_density(),
        secondary_volume_data=sim_state.get_view_into_potential(),
    )
    measurement_tools.animation_writer_3D = animation.AnimationWriter(
        "output/probability_density_time_development_3D.mp4"
    )
    measurement_tools.animation_writer_per_axis = animation.AnimationWriter(
        "output/probability_density_time_development_per_axis.mp4"
    )
    measurement_tools.measurement_plane = measurement.MeasurementPlane(
        wave_tensor=sim_state.wave_tensor,
        delta_x=sim_state.delta_x_bohr_radii,
        location_bohr_radii=28.0,
        simulated_box_width=sim_state.simulated_volume_width_bohr_radii,
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

    # Setup "per axis" probability density:
    measurement_tools.x_axis_probability_density = measurement.ProjectedMeasurement(
        N=sim_state.N, sum_axis=(1, 2), label="X axis"
    )
    measurement_tools.y_axis_probability_density = measurement.ProjectedMeasurement(
        N=sim_state.N, sum_axis=(0, 2), label="Y axis"
    )
    measurement_tools.z_axis_probability_density = measurement.ProjectedMeasurement(
        N=sim_state.N, sum_axis=(0, 1), label="Z axis"
    )

    # Run simulation
    sim_state, measurement_tools, iter_data = core_sim.run_iteration(
        sim_state, measurement_tools
    )

    # Finishing steps:
    measurement_tools.animation_writer_3D.finish()
    measurement_tools.animation_writer_per_axis.finish()
    print("Simulation has finished.")
    print(f"Total simulation time: {iter_data.elapsed_system_time_s} s")
    print(f"Average iteration time: {iter_data.average_iteration_system_time_s} s")
    text_writer.append_iter_data(iter_data)


if __name__ == "__main__":
    sim()
