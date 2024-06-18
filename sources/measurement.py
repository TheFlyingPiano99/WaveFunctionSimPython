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
from typing import Dict
from colorama import Fore, Style
import cupy as cp
import sources.plot as plot
from pathlib import Path

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


class VolumeProbability:
    __name: str
    __bottom_corner_bohr_radii_3: np.array
    __top_corner_bohr_radii_3: np.array
    __enable_image: bool
    __probability_evolution: np.array = np.empty(shape=0, dtype=np.float64)

    def __init__(
        self,
        config: Dict
    ):
        self.__name = try_read_param(config, "name", "Volume probability", "measurement.volume_probabilities")
        self.__bottom_corner_bohr_radii_3 = np.array(
            try_read_param(config, "bottom_corner_bohr_radii_3", [-10.0, -10.0, -10.0], "measurement.volume_probabilities")
        )
        self.__top_corner_bohr_radii_3 = np.array(
            try_read_param(config, "top_corner_bohr_radii_3", [10.0, 10.0, 10.0], "measurement.volume_probabilities")
        )
        for i in range(3):  # flip coordinates between bottom and top if in wrong order
            if self.__bottom_corner_bohr_radii_3[i] > self.__top_corner_bohr_radii_3[i]:
                temp = self.__bottom_corner_bohr_radii_3[i]
                self.__bottom_corner_bohr_radii_3[i] = self.__top_corner_bohr_radii_3[i]
                self.__top_corner_bohr_radii_3[i] = temp
        self.__enable_image = try_read_param(config, "enable_image", True, "measurement.volume_probabilities")

    def get_name(self):
        return self.__name

    def set_name(self, n: str):
        self.__name = n

    def is_enable_image(self):
        return self.__enable_image

    def integrate_probability_density(self, sim_state: SimState):
        probability_density = sim_state.get_view_into_probability_density(
            bottom_corner_bohr_radii=self.__bottom_corner_bohr_radii_3,
            top_corner_bohr_radii=self.__top_corner_bohr_radii_3
        )
        dxdydz = (sim_state.get_delta_x_bohr_radii_3()[0]
                  * sim_state.get_delta_x_bohr_radii_3()[1]
                  * sim_state.get_delta_x_bohr_radii_3()[2])
        probability = min(cp.sum(
            probability_density
        ) * dxdydz, 10.0)    # Min is to prevent inf value if the simulation diverges
        self.__probability_evolution = np.append(
            arr=self.__probability_evolution, values=probability
        )

    def get_probability_evolution_with_name(self):
        return self.__probability_evolution, self.__name

    def clear(self):
        self.__probability_evolution = np.empty(shape=0, dtype=np.float64)


class PlaneProbabilityCurrent:
    __name: str
    __center_bohr_radii_3: np.array
    __normal_vector_3: np.array
    __enable_image: np.array
    __kernel: cp.RawKernel
    __size_bohr_radii_2: np.array
    __resolution_2: np.array
    __probability_current_density: cp.ndarray
    __probability_current_evolution: np.array = np.empty(shape=0, dtype=np.float32)
    __integrated_probability_current_evolution: np.array = np.empty(shape=0, dtype=np.float32)

    def __init__(self, config: Dict, sim_state: SimState):
        self.__name = try_read_param(config, "name", "Probability current", "measurement.plane_probability_currents")
        self.__center_bohr_radii_3 = np.array(
            try_read_param(config, "center_bohr_radii_3", "measurement.plane_probability_currents")
        )
        self.__normal_vector_3 = np.array(
            try_read_param(config, "normal_vector_3", "measurement.plane_probability_currents")
        )
        self.__enable_image = try_read_param(config, "enable_image", "measurement.plane_probability_currents")
        self.__size_bohr_radii_2 = np.array(try_read_param(config, "size_bohr_radii_2", [60.0, 60.0], "measurement.plane_probability_currents"))
        self.__resolution_2 = np.array(try_read_param(config, "resolution_2", [512, 512], "measurement.plane_probability_currents"))

        probability_current_density_kernel = (
            Path("sources/cuda_kernels/probability_current_density.cu").read_text().replace("PATH_TO_SOURCES",
                                                                                            os.path.abspath("sources"))
            .replace("T_WF_FLOAT",
                     "double" if sim_state.is_double_precision_calculation() else "float"))
        self.__kernel = cp.RawKernel(
            probability_current_density_kernel,
            "probability_current_density_kernel"
        )
        self.__probability_current_density = cp.zeros(shape=[self.__resolution_2[0], self.__resolution_2[1]], dtype=cp.float32)

    def get_name(self):
        return self.__name

    def set_name(self, n: str):
        self.__name = n

    def is_enable_image(self):
        return self.__enable_image

    def calculate(self, sim_state: SimState):
        grid_size = (self.__resolution_2[0] // 32, self.__resolution_2[1] // 32, 1)
        block_size = (self.__resolution_2[0] // grid_size[0], self.__resolution_2[1] // grid_size[1], 1)
        self.__kernel(
            grid_size,
            block_size,
            (
                sim_state.get_wave_function(),
                self.__probability_current_density,

                cp.float32(sim_state.get_particle_mass()),

                cp.float32(sim_state.get_delta_x_bohr_radii_3()[0]),
                cp.float32(sim_state.get_delta_x_bohr_radii_3()[1]),
                cp.float32(sim_state.get_delta_x_bohr_radii_3()[2]),

                cp.float32(self.__center_bohr_radii_3[0]),
                cp.float32(self.__center_bohr_radii_3[1]),
                cp.float32(self.__center_bohr_radii_3[2]),

                cp.float32(self.__normal_vector_3[0]),
                cp.float32(self.__normal_vector_3[1]),
                cp.float32(self.__normal_vector_3[2]),

                cp.float32(self.__size_bohr_radii_2[0]),
                cp.float32(self.__size_bohr_radii_2[1]),

                cp.uint32(sim_state.get_number_of_voxels_3()[0]),
                cp.uint32(sim_state.get_number_of_voxels_3()[1]),
                cp.uint32(sim_state.get_number_of_voxels_3()[2]),

                cp.float32(sim_state.get_observation_box_bottom_corner_bohr_radii_3()[0]),
                cp.float32(sim_state.get_observation_box_bottom_corner_bohr_radii_3()[1]),
                cp.float32(sim_state.get_observation_box_bottom_corner_bohr_radii_3()[2]),

                cp.float32(sim_state.get_observation_box_top_corner_bohr_radii_3()[0]),
                cp.float32(sim_state.get_observation_box_top_corner_bohr_radii_3()[1]),
                cp.float32(sim_state.get_observation_box_top_corner_bohr_radii_3()[2])
            )
        )
        dwdh = (self.__size_bohr_radii_2[0] / self.__resolution_2[0]
                * self.__size_bohr_radii_2[1] / self.__resolution_2[1])
        probability_current = cp.sum(self.__probability_current_density) * dwdh

        self.__probability_current_evolution = (
            np.append(arr=self.__probability_current_evolution, values=probability_current))
        self.__integrated_probability_current_evolution = np.append(
            arr=self.__integrated_probability_current_evolution,
            values=np.sum(self.__probability_current_evolution) * sim_state.get_delta_time_h_bar_per_hartree()
        )

    def get_probability_current_evolution_with_name(self):
        return self.__probability_current_evolution, self.__name

    def get_integrated_probability_current_evolution_with_name(self):
        return self.__integrated_probability_current_evolution, self.__name


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
    __volumetric_image_capture_interval: int
    __volumetric_animation_capture_interval: int
    __animation_writer_volumetric: animation.AnimationWriter = None
    __animation_writer_per_axis: animation.AnimationWriter = None
    __x_axis_probability_density: ProjectedMeasurement = None
    __y_axis_probability_density: ProjectedMeasurement = None
    __z_axis_probability_density: ProjectedMeasurement = None
    __projected_potential: ProjectedMeasurement = None
    __enable_per_axis_image = False
    __enable_per_axis_animation = False
    __per_axis_image_capture_interval: int = 50
    __volume_probabilities: list[VolumeProbability] = []
    __plane_probability_currents: list[PlaneProbabilityCurrent] = []
    __show_figures: bool = True

    def _resolve_naming_conflicts(self, list, new_item):
        for item in list:
            if (item.get_name() == new_item.get_name()):
                print(
                    Fore.RED + f"Found multiple measurement tools of the same type with equal name: \"{new_item.get_name()}\"" + Style.RESET_ALL)
                index = 1
                unique_name = False
                while not unique_name:
                    unique_name = True
                    for item2 in list:
                        if (new_item.get_name() + f"_{index}") == item2.get_name():
                            unique_name = False
                            index += 1
                            break
                new_item.set_name(new_item.get_name() + f"_{index}")
                print(Fore.RED + f"Renamed one to \"{new_item.get_name()}\"." + Style.RESET_ALL)
                break


    def __init__(self, config: Dict, sim_state: SimState):
        # Volumetric visualization:
        self.__enable_volumetric_animation = try_read_param(config, "measurement.volumetric.enable_animation", False)
        self.__enable_volumetric_image = try_read_param(config, "measurement.volumetric.enable_image", False)
        self.__volumetric_image_capture_interval = try_read_param(config, "measurement.volumetric.image_capture_iteration_interval", 100)
        self.__volumetric_animation_capture_interval = try_read_param(config, "measurement.volumetric.animation_frame_capture_iteration_interval", 5)
        if self.__enable_volumetric_image or self.__enable_volumetric_animation:
            self.__volumetric = VolumetricVisualization(
                wave_function=sim_state.get_view_into_wave_function(),
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

        # Volume probability:
        volume_confs = try_read_param(config, "measurement.volume_probabilities", [])
        for conf in volume_confs:
            new_v = VolumeProbability(conf)
            self._resolve_naming_conflicts(self.__volume_probabilities, new_v)
            self.__volume_probabilities.append(new_v)

        plane_confs = try_read_param(config, "measurement.plane_probability_currents", [])
        for conf in plane_confs:
            new_p = PlaneProbabilityCurrent(conf, sim_state)
            self._resolve_naming_conflicts(self.__plane_probability_currents, new_p)
            self.__plane_probability_currents.append(new_p)

    def measure_and_render(self, sim_state: SimState, iter_data: IterData):
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
                wave_function=sim_state.get_view_into_wave_function(),
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

        for volume in self.__volume_probabilities:
            if volume.is_enable_image():
                volume.integrate_probability_density(sim_state)

        for plane in self.__plane_probability_currents:
            if plane.is_enable_image():
                plane.calculate(sim_state)


    def finish(self, sim_state: SimState):
        if self.__enable_volumetric_animation:
            self.__animation_writer_volumetric.finish()
        if self.__enable_per_axis_animation:
            self.__animation_writer_per_axis.finish()

        volume_probability_evolutions = []
        for v in self.__volume_probabilities:
            if v.is_enable_image():
                volume_probability_evolutions.append(v.get_probability_evolution_with_name())
        if len(volume_probability_evolutions) > 0:
            plot.plot_probability_evolution(
                out_dir=sim_state.get_output_dir(),
                file_name="volume_probability_evolution.png",
                title="Volume probability evolution",
                y_label="Probability",
                probability_evolutions=volume_probability_evolutions,
                delta_t=sim_state.get_delta_time_h_bar_per_hartree(),
                show_fig=self.__show_figures
            )

        probability_current_evolutions = []
        for p in self.__plane_probability_currents:
            if p.is_enable_image():
                probability_current_evolutions.append(p.get_probability_current_evolution_with_name())
        if len(probability_current_evolutions) > 0:
            plot.plot_probability_evolution(
                out_dir=sim_state.get_output_dir(),
                file_name="probability_current_evolution.png",
                title="Probability current evolution",
                y_label="Probability current",
                probability_evolutions=probability_current_evolutions,
                delta_t=sim_state.get_delta_time_h_bar_per_hartree(),
                show_fig=self.__show_figures,
                y_min = -1.0,
                y_max = 1.0
            )

        integrated_probability_current_evolutions = []
        for p in self.__plane_probability_currents:
            if p.is_enable_image():
                integrated_probability_current_evolutions.append(
                    p.get_integrated_probability_current_evolution_with_name()
                )
        if len(integrated_probability_current_evolutions) > 0:
            plot.plot_probability_evolution(
                out_dir=sim_state.get_output_dir(),
                file_name="integrated_probability_current_evolution.png",
                title="Integrated probability current evolution",
                y_label="Probability",
                probability_evolutions=integrated_probability_current_evolutions,
                delta_t=sim_state.get_delta_time_h_bar_per_hartree(),
                show_fig=self.__show_figures,
                y_min=-1.0,
                y_max=1.0
            )
