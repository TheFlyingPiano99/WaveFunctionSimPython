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


def sim():
    print(text_writer.get_title_text())
    print("\n")
    sim_state = init.initialize()

    # Check display availability
    have_display = "DISPLAY" in os.environ
    if not have_display:
        exitval = os.system('python -c "import matplotlib.pyplot as plt; plt.figure()"')
        have_display = (exitval == 0)
    """
    if have_display:
        print("Display connected.\n")
    else:
        print("Display not connected.\n")
    """

    if sim_state.enable_visual_output:
        print(Fore.BLUE + "Visual output enabled." + Style.RESET_ALL)
    else:
        print(Fore.BLUE + "Visual output disabled." + Style.RESET_ALL)

    if not have_display and sim_state.enable_visual_output:    # Create virtual display if no physical connected
        from xvfbwrapper import Xvfb
        vdisplay = Xvfb()
        vdisplay.start()
        print("Created virtual display.\n")

    text_writer.write_sim_state(sim_state)

    print(
        "\n****************************************************************************\n"
    )

    measurement_tools = measurement.MeasurementTools()
    if sim_state.enable_visual_output:
        measurement_tools = measurement.MeasurementTools()
        measurement_tools.volumetric = volume_visualization.VolumetricVisualization(
            probability=sim_state.get_view_into_raw_wave_function(),
            potential=sim_state.get_view_into_complex_potential(),
            coulomb_potential=sim_state.get_view_into_coulomb_potential(),
            cam_rotation_speed=sim_state.config["view"]["volumetric"]["camera_rotation_speed"],
            azimuth=sim_state.config["view"]["volumetric"]["camera_azimuth"],
        )
        measurement_tools.volumetric.elevation = sim_state.config["view"]["volumetric"]["camera_elevation"]
        measurement_tools.volumetric.set_light_direction(sim_state.config["view"]["volumetric"]["light_direction"])
        measurement_tools.volumetric.light_rotation_speed = sim_state.config["view"]["volumetric"]["light_rotation_speed"]
        measurement_tools.volumetric.light_elevation_speed = sim_state.config["view"]["volumetric"]["light_elevation_speed"]

        measurement_tools.animation_writer_3D = animation.AnimationWriter(
            os.path.join(sim_state.output_dir, "probability_density_time_development_3D.mp4")
        )
        measurement_tools.animation_writer_per_axis = animation.AnimationWriter(
            os.path.join(sim_state.output_dir, "probability_density_time_development_per_axis.mp4")
        )
        measurement_tools.measurement_plane = measurement.MeasurementPlane(
            delta_x=sim_state.delta_x_bohr_radii,
            location_bohr_radii=30.0,
            simulated_box_width=sim_state.simulated_volume_width_bohr_radii,
            viewing_window_bottom_voxel=sim_state.viewing_window_bottom_corner_voxel,
            viewing_window_top_voxel=sim_state.viewing_window_top_corner_voxel,
        )
        measurement_tools.measurement_volume_full = measurement.AAMeasurementVolume(
            bottom_corner=sim_state.viewing_window_bottom_corner_voxel,
            top_corner=sim_state.viewing_window_top_corner_voxel,
            label="Full volume",
        )
        '''
        measurement_tools.measurement_volume_first_half = measurement.AAMeasurementVolume(
            bottom_corner=sim_state.viewing_window_bottom_corner_voxel,
            top_corner=np.array(
                sim_state.viewing_window_top_corner_voxel
                - (
                    sim_state.viewing_window_top_corner_voxel
                    - sim_state.viewing_window_bottom_corner_voxel
                )
                * np.array([0, 0.5, 0]),
                dtype=int,
            ),
            label="Substrate",
        )
        measurement_tools.measurement_volume_second_half = measurement.AAMeasurementVolume(
            bottom_corner=np.array(
                sim_state.viewing_window_bottom_corner_voxel
                + (
                    sim_state.viewing_window_top_corner_voxel
                    - sim_state.viewing_window_bottom_corner_voxel
                )
                * np.array([0, 0.5, 0]),
                dtype=int,
            ),
            top_corner=sim_state.viewing_window_top_corner_voxel,
            label="Floating gate",
        )
        '''

        # Setup "per axis" probability density:
        measurement_tools.x_axis_probability_density = measurement.ProjectedMeasurement(
            min_voxel=sim_state.viewing_window_bottom_corner_voxel[0],
            max_voxel=sim_state.viewing_window_top_corner_voxel[0],
            near_voxel=sim_state.viewing_window_bottom_corner_voxel[1],
            far_voxel=sim_state.viewing_window_top_corner_voxel[1],
            left_edge=sim_state.viewing_window_bottom_corner_bohr_radii[0],
            right_edge=sim_state.viewing_window_top_corner_bohr_radii[0],
            sum_axis=(1, 2),
            label=sim_state.config["view"]["per_axis_plot"]["x_axis_label"],
        )
        measurement_tools.y_axis_probability_density = measurement.ProjectedMeasurement(
            min_voxel=sim_state.viewing_window_bottom_corner_voxel[1],
            max_voxel=sim_state.viewing_window_top_corner_voxel[1],
            near_voxel=sim_state.viewing_window_bottom_corner_voxel[2],
            far_voxel=sim_state.viewing_window_top_corner_voxel[2],
            left_edge=sim_state.viewing_window_bottom_corner_bohr_radii[1],
            right_edge=sim_state.viewing_window_top_corner_bohr_radii[1],
            sum_axis=(0, 2),
            label=sim_state.config["view"]["per_axis_plot"]["y_axis_label"],
        )
        measurement_tools.z_axis_probability_density = measurement.ProjectedMeasurement(
            min_voxel=sim_state.viewing_window_bottom_corner_voxel[2],
            max_voxel=sim_state.viewing_window_top_corner_voxel[2],
            near_voxel=sim_state.viewing_window_bottom_corner_voxel[0],
            far_voxel=sim_state.viewing_window_top_corner_voxel[0],
            left_edge=sim_state.viewing_window_bottom_corner_bohr_radii[2],
            right_edge=sim_state.viewing_window_top_corner_bohr_radii[2],
            sum_axis=(0, 1),
            label=sim_state.config["view"]["per_axis_plot"]["z_axis_label"],
        )
        measurement_tools.projected_potential = measurement.ProjectedMeasurement(
            min_voxel=sim_state.viewing_window_bottom_corner_voxel[0],
            max_voxel=sim_state.viewing_window_top_corner_voxel[0],
            near_voxel=sim_state.N // 2,
            far_voxel=sim_state.N // 2 + 1,
            left_edge=sim_state.viewing_window_bottom_corner_bohr_radii[0],
            right_edge=sim_state.viewing_window_top_corner_bohr_radii[0],
            sum_axis=(1, 2),
            label=sim_state.config["view"]["per_axis_plot"]["potential_label"],
        )
        #measurement_tools.projected_potential.scale_factor = sim_state.config["view"]["per_axis_plot"]["potential_plot_scale"]
        measurement_tools.projected_potential.scale_factor = (0.20 / sim_state.get_view_into_potential().max())
        #measurement_tools.projected_potential.offset = sim_state.config["view"]["per_axis_plot"]["potential_plot_offset"]
        measurement_tools.projected_potential.offset = 0.0

        # Run simulation
    sim_state, measurement_tools, iter_data = core_sim.run_iteration(
        sim_state, measurement_tools
    )

    # Finishing steps:
    if sim_state.enable_visual_output:
        measurement_tools.animation_writer_3D.finish()
        measurement_tools.animation_writer_per_axis.finish()
    print(Fore.GREEN + "Simulation has finished." + Style.RESET_ALL)
    print(text_writer.get_finish_text(iter_data))
    text_writer.append_iter_data(iter_data, sim_state)

    if not have_display and sim_state.enable_visual_output:
        vdisplay.stop()


if __name__ == "__main__":
    sim()
