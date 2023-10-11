from sources import volume_visualization, animation, measurement, text_writer
import sources.initialisation as init
import sources.core_sim as core_sim
import sys
import os
import signal
import sources.text_writer as text_writer
from colorama import Fore, Style
import cupy as cp
import numpy as np


class MeasurementTools:
    measurement_plane: measurement.MeasurementPlane
    canvas: volume_visualization.VolumeCanvas
    animation_writer_3D: animation.AnimationWriter
    animation_writer_per_axis: animation.AnimationWriter
    x_axis_probability_density: measurement.ProjectedMeasurement
    y_axis_probability_density: measurement.ProjectedMeasurement
    z_axis_probability_density: measurement.ProjectedMeasurement


def sim():
    print(text_writer.get_title_text())

    out_dir = "output/"
    if os.path.exists(out_dir):
        if os.listdir(out_dir):
            answer = ""
            while not answer in {"y", "n"}:
                print(
                    "Output directory is not empty.\n"
                    'Continuing will override previous files under "./output/".\n'
                    "Would you still like to continue [y/n]?",
                    end=" ",
                )
                answer = input()
                if answer == "n":
                    print("Exiting application.")
                    sys.exit(0)
    print("\n")
    sim_state = init.initialize()
    text_writer.write_sim_state(sim_state)

    print(
        "****************************************************************************"
    )
    measurement_tools = MeasurementTools()
    measurement_tools.canvas = volume_visualization.VolumeCanvas(
        volume_data=sim_state.get_view_into_probability_density(),
        secondary_volume_data=sim_state.get_view_into_potential(),
        cam_rotation_speed=sim_state.config["View"]["camera_rotation_speed"],
        azimuth=sim_state.config["View"]["camera_azimuth"],
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
        viewing_window_bottom_voxel=sim_state.viewing_window_bottom_corner_voxel,
        viewing_window_top_voxel=sim_state.viewing_window_top_corner_voxel,
    )
    measurement_tools.measurement_volume_full = measurement.AAMeasurementVolume(
        bottom_corner=sim_state.viewing_window_bottom_corner_voxel,
        top_corner=sim_state.viewing_window_top_corner_voxel,
        label="Full volume",
    )
    measurement_tools.measurement_volume_first_half = measurement.AAMeasurementVolume(
        bottom_corner=sim_state.viewing_window_bottom_corner_voxel,
        top_corner=np.array(
            sim_state.viewing_window_top_corner_voxel
            - (
                sim_state.viewing_window_top_corner_voxel
                - sim_state.viewing_window_bottom_corner_voxel
            )
            * np.array([0.5, 0, 0]),
            dtype=int,
        ),
        label="First half",
    )
    measurement_tools.measurement_volume_second_half = measurement.AAMeasurementVolume(
        bottom_corner=np.array(
            sim_state.viewing_window_bottom_corner_voxel
            + (
                sim_state.viewing_window_top_corner_voxel
                - sim_state.viewing_window_bottom_corner_voxel
            )
            * np.array([0.5, 0, 0]),
            dtype=int,
        ),
        top_corner=sim_state.viewing_window_top_corner_voxel,
        label="Second half",
    )

    # Setup "per axis" probability density:
    measurement_tools.x_axis_probability_density = measurement.ProjectedMeasurement(
        min_voxel=sim_state.viewing_window_bottom_corner_voxel[0],
        max_voxel=sim_state.viewing_window_top_corner_voxel[0],
        left_edge=sim_state.viewing_window_bottom_corner_bohr_radii[0],
        right_edge=sim_state.viewing_window_top_corner_bohr_radii[0],
        sum_axis=(1, 2),
        label="X axis",
    )
    measurement_tools.y_axis_probability_density = measurement.ProjectedMeasurement(
        min_voxel=sim_state.viewing_window_bottom_corner_voxel[1],
        max_voxel=sim_state.viewing_window_top_corner_voxel[1],
        left_edge=sim_state.viewing_window_bottom_corner_bohr_radii[1],
        right_edge=sim_state.viewing_window_top_corner_bohr_radii[1],
        sum_axis=(0, 2),
        label="Y axis",
    )
    measurement_tools.z_axis_probability_density = measurement.ProjectedMeasurement(
        min_voxel=sim_state.viewing_window_bottom_corner_voxel[2],
        max_voxel=sim_state.viewing_window_top_corner_voxel[2],
        left_edge=sim_state.viewing_window_bottom_corner_bohr_radii[2],
        right_edge=sim_state.viewing_window_top_corner_bohr_radii[2],
        sum_axis=(0, 1),
        label="Z axis",
    )

    # Run simulation
    sim_state, measurement_tools, iter_data = core_sim.run_iteration(
        sim_state, measurement_tools
    )

    # Finishing steps:
    measurement_tools.animation_writer_3D.finish()
    measurement_tools.animation_writer_per_axis.finish()
    print(Fore.GREEN + "Simulation has finished." + Style.RESET_ALL)
    print(text_writer.get_finish_text(iter_data))
    text_writer.append_iter_data(iter_data)


if __name__ == "__main__":
    sim()
