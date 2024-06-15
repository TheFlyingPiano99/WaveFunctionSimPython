import numpy as np
from numba.experimental import jitclass
from numba import types
import numba
from sources.volumetric_visualization import VolumetricVisualization
import sources.animation as animation
from sources.sim_state import SimState
from sources.config_read_helper import try_read_param
from sources.iter_data import IterData
import os

class MeasurementPlane:
    def __init__(
        self,
        delta_x_3: np.array,
        location_bohr_radii: float,
        simulated_box_dimensions_3: float,
        viewing_window_bottom_voxel_3: np.array,
        viewing_window_top_voxel_3: np.array,
    ):
        self.plane_dwell_time_density = np.zeros(
            shape=(
                viewing_window_top_voxel_3[2] - viewing_window_bottom_voxel_3[2],
                viewing_window_top_voxel_3[1] - viewing_window_bottom_voxel_3[1],
            )
        )
        self.plane_probability_density = np.zeros(
            shape=(
                viewing_window_top_voxel_3[2] - viewing_window_bottom_voxel_3[2],
                viewing_window_top_voxel_3[1] - viewing_window_bottom_voxel_3[1],
            )
        )
        self.cumulated_time = 0.0
        self.voxel_x = int((location_bohr_radii + simulated_box_dimensions_3[0] * 0.5) / delta_x_3[0])
        if self.voxel_x > viewing_window_top_voxel_3[0] - 1:
            self.voxel_x = viewing_window_top_voxel_3[0] - 1
        elif self.voxel_x < viewing_window_bottom_voxel_3[0]:
            self.voxel_x = viewing_window_bottom_voxel_3[0]
        self.viewing_window_bottom_voxel = viewing_window_bottom_voxel_3
        self.viewing_window_top_voxel = viewing_window_top_voxel_3

    def integrate(self, probability_density, delta_time):
        self.plane_probability_density = probability_density[
            self.voxel_x,
            self.viewing_window_bottom_voxel[1] : self.viewing_window_top_voxel[1],
            self.viewing_window_bottom_voxel[2] : self.viewing_window_top_voxel[2],
        ]
        self.plane_dwell_time_density += self.plane_probability_density * delta_time
        self.cumulated_time += delta_time

    def get_probability_density(self):
        return self.plane_probability_density

    def get_dwell_time(self):
        return self.plane_dwell_time_density


class AAMeasurementVolume:
    def __init__(
        self,
        bottom_corner : np.array,
        top_corner: np.array,
        label : str,
    ):
        self.top_corner = top_corner
        self.bottom_corner = bottom_corner
        self.label = label
        self.probability = 0.0
        self.probability_evolution = np.empty(shape=0, dtype=np.float64)

    def integrate_probability_density(self, probability_density):
        self.probability = min(np.sum(
            probability_density[
                self.bottom_corner[0] : self.top_corner[0],
                self.bottom_corner[1] : self.top_corner[1],
                self.bottom_corner[2] : self.top_corner[2],
            ]
        ), 10.0)
        self.probability_evolution = np.append(
            arr=self.probability_evolution, values=self.probability
        )

    def clear(self):
        self.probability = 0.0
        self.probability_evolution = np.empty(shape=0, dtype=np.float64)

    def get_probability(self):
        return self.probability

    def get_probability_evolution(self):
        return self.probability_evolution, self.label


"""
For calculating the probability density per axis
without any assuptions about the other dimensions.
"""


class ProjectedMeasurement:
    probability_density: np.array
    min_voxel: int
    max_voxel: int
    near_voxel: int     # in directions of summing axes
    far_voxel: int      # in directions of summing axes
    left_edge_bohr_radii: float
    right_edge_boh_radii: float
    sum_axis: tuple
    label: str
    scale_factor: float = 1.0
    offset: float = 0.0

    def __init__(self,
                 min_voxel : int,
                 max_voxel : int,
                 left_edge : float,
                 right_edge : float,
                 sum_axis : tuple,
                 label : str,
                 near_voxel : int = None,
                 far_voxel : int = None,
                 ):
        self.min_voxel = min_voxel
        self.max_voxel = max_voxel

        if self.min_voxel > self.max_voxel:
            temp = self.min_voxel
            self.min_voxel = self.max_voxel
            self.max_voxel = temp
        self.probability_density = np.zeros(
            shape=(max_voxel - min_voxel), dtype=np.float64
        )

        self.label = label
        self.left_edge_bohr_radii = left_edge
        self.right_edge_bohr_radii = right_edge
        self.sum_axis = sum_axis

        if near_voxel is None:
            self.near_voxel = self.min_voxel     # let's assume cube shaped viewing window
        else:
            self.near_voxel = near_voxel
        if far_voxel is None:
            self.far_voxel = self.max_voxel     # let's assume cube shaped viewing window
        else:
            self.far_voxel = far_voxel

    def integrate_probability_density(self, probability_density_tensor: np.ndarray):
        near_far = []
        for i in range(3):
            if i in self.sum_axis:
                near_far.append([self.near_voxel, self.far_voxel])
            else:
                near_far.append([self.min_voxel, self.max_voxel])

        self.probability_density = np.sum(
            a=probability_density_tensor[
                near_far[0][0]: near_far[0][1],
                near_far[1][0]: near_far[1][1],
                near_far[2][0]: near_far[2][1],
              ], axis=self.sum_axis)

    def get_probability_density_with_label(self):
        return (
            self.offset + self.probability_density * self.scale_factor,
            self.label,
            self.left_edge_bohr_radii,
            self.right_edge_bohr_radii,
            self.scale_factor
        )


class MeasurementTools:
    __volumetric: VolumetricVisualization = None
    __enable_volumetric_image = False
    __enable_volumetric_animation = False
    __volumetric_image_capture_interval: int = 50
    __animation_writer_volumetric: animation.AnimationWriter = None
    __animation_writer_per_axis: animation.AnimationWriter = None
    __x_axis_probability_density: ProjectedMeasurement = None
    __y_axis_probability_density: ProjectedMeasurement = None
    __z_axis_probability_density: ProjectedMeasurement = None
    __projected_potential: ProjectedMeasurement = None
    __enable_per_axis_image = False
    __enable_per_axis_animation = False
    __per_axis_image_capture_interval: int = 50


    def __init__(self, config, sim_state: SimState):
        # Volumetric visualization:
        self.__enable_volumetric_animation = try_read_param(config, "measurement.volumetric.enable_animation", False)
        self.__enable_volumetric_image = try_read_param(config, "measurement.volumetric.enable_image", False)
        if self.__enable_volumetric_image or self.__enable_volumetric_animation:
            self.__volumetric = VolumetricVisualization(
                wave_function=sim_state.get_view_into_raw_wave_function(),
                potential=sim_state.get_view_into_complex_potential(),
                coulomb_potential=sim_state.get_view_into_coulomb_potential(),
                cam_rotation_speed=try_read_param(config, "measurement.volumetric.camera_rotation_speed", 0.0),
                azimuth=try_read_param(config, "measurement.volumetric.camera_azimuth", 0.0)
            )
        if self.__enable_volumetric_animation:
            self.__animation_writer_volumetric = animation.AnimationWriter(
                os.path.join(sim_state.get_output_dir(), "volumetric_visualization.mp4"),
                try_read_param(config, "measurement.volumetric.animation_frame_rate", 25),
                try_read_param(config, "measurement.volumetric.animation_frame_capture_iteration_interval", 1)
            )

        # "per axis" probability density:
        self.__enable_per_axis_animation = try_read_param(config, "measurement.per_axis_plot.enable_animation", False)
        self.__enable_per_axis_image = try_read_param(config, "measurement.per_axis_plot.enable_image", False)
        if self.__enable_per_axis_image or self.__enable_per_axis_animation:
            self.__x_axis_probability_density = ProjectedMeasurement(
                min_voxel=sim_state.get_observation_box_bottom_corner_voxel_3()[0],
                max_voxel=sim_state.get_observation_box_top_corner_voxel_3()[0],
                near_voxel=sim_state.get_observation_box_bottom_corner_voxel_3()[1],
                far_voxel=sim_state.get_observation_box_top_corner_voxel_3()[1],
                left_edge=sim_state.get_observation_box_bottom_corner_bohr_radii_3()[0],
                right_edge=sim_state.get_observation_box_top_corner_bohr_radii_3()[0],
                sum_axis=(1, 2),
                label=try_read_param(config, "measurement.per_axis_plot.x_axis_label", "X axis")
            )
            self.__y_axis_probability_density = ProjectedMeasurement(
                min_voxel=sim_state.get_observation_box_bottom_corner_voxel_3()[1],
                max_voxel=sim_state.get_observation_box_top_corner_voxel_3()[1],
                near_voxel=sim_state.get_observation_box_bottom_corner_voxel_3()[2],
                far_voxel=sim_state.get_observation_box_top_corner_voxel_3()[2],
                left_edge=sim_state.get_observation_box_bottom_corner_bohr_radii_3()[1],
                right_edge=sim_state.get_observation_box_top_corner_bohr_radii_3()[1],
                sum_axis=(0, 2),
                label=try_read_param(config, "measurement.per_axis_plot.y_axis_label", "Y axis"),
            )
            self.__z_axis_probability_density = ProjectedMeasurement(
                min_voxel=sim_state.get_observation_box_bottom_corner_voxel_3()[2],
                max_voxel=sim_state.get_observation_box_top_corner_voxel_3()[2],
                near_voxel=sim_state.get_observation_box_bottom_corner_voxel_3()[0],
                far_voxel=sim_state.get_observation_box_top_corner_voxel_3()[0],
                left_edge=sim_state.get_observation_box_bottom_corner_bohr_radii_3()[2],
                right_edge=sim_state.get_observation_box_top_corner_bohr_radii_3()[2],
                sum_axis=(0, 1),
                label=try_read_param(config, "measurement.per_axis_plot.z_axis_label", "Z axis"),
            )
            self.__projected_potential = ProjectedMeasurement(
                min_voxel=sim_state.get_observation_box_top_corner_voxel_3()[0],
                max_voxel=sim_state.get_observation_box_top_corner_voxel_3()[0],
                near_voxel=sim_state.get_number_of_voxels_3()[0] // 2,
                far_voxel=sim_state.get_number_of_voxels_3()[0] // 2 + 1,
                left_edge=sim_state.get_observation_box_bottom_corner_bohr_radii_3()[0],
                right_edge=sim_state.get_observation_box_top_corner_bohr_radii_3()[0],
                sum_axis=(1, 2),
                label=try_read_param(config, "measurement.per_axis_plot.potential_label", "Potential"),
            )
            self.__projected_potential.scale_factor = (0.20 / sim_state.get_view_into_potential().max())
            self.__projected_potential.offset = 0.0

        if self.__enable_per_axis_animation:
            self.__animation_writer_per_axis = animation.AnimationWriter(
                os.path.join(sim_state.get_output_dir(), "per-axis_visualization.mp4"),
                try_read_param(config, "measurement.per_axis_plot.animation_frame_rate", 25),
                try_read_param(config, "measurement.per_axis_plot.animation_frame_capture_iteration_interval", 1)
            )

    def measure_and_render(self, sim_state: SimState, iter_data: IterData):
        # Update all measurement tools:
        '''
            measurement_tools.measurement_volume_full.integrate_probability_density(
            sim_state.probability_density
        )
        '''

        '''
        measurement_tools.measurement_volume_first_half.integrate_probability_density(
            sim_state.probability_density
        )
        measurement_tools.measurement_volume_second_half.integrate_probability_density(
            sim_state.probability_density
        )
        '''

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
            measurement_tools.projected_potential.integrate_probability_density(
                np.real(sim_state.localised_potential_to_visualize_hartree)
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
                    measurement_tools.projected_potential.get_probability_density_with_label(),
                ],
                delta_x_3=sim_state.delta_x_bohr_radii_3,
                delta_t=sim_state.delta_time_h_bar_per_hartree,
                potential_scale=sim_state.config["view"]["per_axis_plot"]["potential_plot_scale"],
                index=iter_data.i,
                show_fig=False,
            )
        '''

        # Update the volumetric visualization if needed:
        if (
                (
                        self.__enable_volumetric_animation
                        and iter_data.i % self.__animation_writer_volumetric.frame_capture_iteration_interval == 0
                )
                or
                (
                        self.__enable_volumetric_image
                        and iter_data.i % self.__volumetric_image_capture_interval == 0
                )
        ):
            self.__volumetric.update(
                wave_function=sim_state.get_view_into_raw_wave_function(),
                potential=sim_state.get_view_into_complex_potential(),
                iter_count=iter_data.i,
                delta_time_h_bar_per_hartree=sim_state.get_delta_time_h_bar_per_hartree(),
            )
        # Append animation frame:
        if (
                self.__enable_volumetric_animation
                and iter_data.i % self.__animation_writer_volumetric.frame_capture_iteration_interval == 0
        ):
            self.__animation_writer_volumetric.add_frame(
                img=self.__volumetric.render()
            )
        # Save image:
        if (
                self.__enable_volumetric_image
                and iter_data.i % self.__volumetric_image_capture_interval == 0
        ):
            self.__volumetric.render_to_png(out_dir=sim_state.get_output_dir(), index=iter_data.i)

        """
        if iter_data.i % iter_data.animation_frame_step_interval == 0:
            measurement_tools.animation_writer_3D.add_frame(
                measurement_tools.volumetric.render()
            )
            measurement_tools.animation_writer_per_axis.add_frame(
                measurement_tools.per_axis_density_plot
            )
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
                delta_x_3=sim_state.delta_x_bohr_radii_3,
                delta_t=sim_state.delta_time_h_bar_per_hartree
            )
        """

    def finish(self):
        self.__animation_writer_volumetric.finish()
        self.__animation_writer_per_axis.finish()
