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
import sources.math_utils as math_utils


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
                                         self.viewing_window_bottom_voxel[1]: self.viewing_window_top_voxel[1],
                                         self.viewing_window_bottom_voxel[2]: self.viewing_window_top_voxel[2],
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
    __volume_probability_kernel: cp.RawKernel = None
    __probability_buffer: cp.array
    __kernel_grid_size: np.array
    __kernel_block_size: np.array
    __bottom_voxel: np.array
    __voxel_count: np.array
    __cuda_stream: cp.cuda.Stream

    def __init__(
            self,
            config: Dict,
            sim_state: SimState
    ):
        self.__name = try_read_param(config, "name", "Volume probability", "measurement.volume_probabilities")
        self.__bottom_corner_bohr_radii_3 = np.array(
            try_read_param(config, "bottom_corner_bohr_radii_3", [-10.0, -10.0, -10.0], "measurement.volume_probabilities")
        )
        self.__top_corner_bohr_radii_3 = np.array(
            try_read_param(config, "top_corner_bohr_radii_3", [10.0, 10.0, 10.0], "measurement.volume_probabilities")
        )
        for i in range(3):
            # flip coordinates between bottom and top if in wrong order
            if self.__bottom_corner_bohr_radii_3[i] > self.__top_corner_bohr_radii_3[i]:
                temp = self.__bottom_corner_bohr_radii_3[i]
                self.__bottom_corner_bohr_radii_3[i] = self.__top_corner_bohr_radii_3[i]
                self.__top_corner_bohr_radii_3[i] = temp

        self.__enable_image = try_read_param(config, "enable_image", True, "measurement.volume_probabilities")

        self.__bottom_voxel = sim_state.transform_physical_coordinate_to_voxel_3(self.__bottom_corner_bohr_radii_3)
        top_voxel = sim_state.transform_physical_coordinate_to_voxel_3(self.__top_corner_bohr_radii_3)
        top_voxel += np.array([1, 1, 1], dtype=top_voxel.dtype)  # Add because of the inclusive boundary requirement of the Simpson scheme
        volume_probability_kernel_source = (Path("sources/cuda_kernels/volume_probability.cu")
                                            .read_text().replace("PATH_TO_SOURCES", os.path.abspath("sources"))
                                            .replace("T_WF_FLOAT",
                                                     "double" if sim_state.is_double_precision() else "float"))

        self.__voxel_count = top_voxel - self.__bottom_voxel
        for i in range(3):
            # Offset boundary if even voxels are included. (Because of the Simpson integration scheme.)
            if (self.__voxel_count[i] % 2 == 0):
                print(
                    Fore.RED + f"Volume probability \"{self.__name}\": truncating volume probability area along {i}. axis!" + Style.RESET_ALL)
                self.__voxel_count[i] -= 1
                top_voxel[i] -= 1
        kernel_shape = (
            math_utils.nearest_power_of_2(self.__voxel_count[0]),
            math_utils.nearest_power_of_2(self.__voxel_count[1]),
            math_utils.nearest_power_of_2(self.__voxel_count[2]),
        )
        self.__kernel_grid_size, self.__kernel_block_size = math_utils.get_grid_size_block_size(kernel_shape, reduced_thread_count=True)
        func_name = f"volume_probability_kernel< {self.__voxel_count[0]}, {self.__voxel_count[1]}, {self.__voxel_count[2]} >"
        self.__volume_probability_kernel = cp.RawModule(
            code=volume_probability_kernel_source,
            name_expressions=[func_name],
        ).get_function(func_name)
        dtype = cp.float64 if sim_state.is_double_precision() else cp.float32
        self.__probability_buffer = cp.array([0.0], dtype=dtype)
        if self.__enable_image:
            print(
                f"{self.__name} bottom voxel (included in calc.): ({self.__bottom_voxel[0]}, {self.__bottom_voxel[1]}, {self.__bottom_voxel[2]})")
            print(f"{self.__name} top voxel (included in calc.): ({top_voxel[0] - 1}, {top_voxel[1] - 1}, {top_voxel[2] - 1})")
            print(
                f"{self.__name} integral voxel count: {self.__kernel_grid_size[0] * self.__kernel_block_size[0]}, {self.__kernel_grid_size[1] * self.__kernel_block_size[1]}, {self.__kernel_grid_size[2] * self.__kernel_block_size[2]}"
            )
            bottom_r = sim_state.transform_voxel_to_physical_coordinate_3(self.__bottom_voxel)
            top_r = sim_state.transform_voxel_to_physical_coordinate_3(top_voxel)
            print(f"{self.__name} integral bottom pos: ({bottom_r[0]}, {bottom_r[1]}, {bottom_r[2]}) Bohr radii")
            print(f"{self.__name} integral top pos: ({top_r[0]}, {top_r[1]}, {top_r[2]}) Bohr radii")
        self.__cuda_stream = cp.cuda.Stream()

    def get_name(self):
        return self.__name

    def set_name(self, n: str):
        self.__name = n

    def is_enable_image(self):
        return self.__enable_image

    def calculate(self, sim_state: SimState):
        with self.__cuda_stream:
            wave_function = sim_state.get_wave_function()
            delta_r = sim_state.get_delta_x_bohr_radii_3()
            N = sim_state.get_number_of_voxels_3()
            dp = sim_state.is_double_precision()
            self.__probability_buffer[0] = 0.0
            self.__volume_probability_kernel(
                self.__kernel_grid_size,
                self.__kernel_block_size,
                (
                    wave_function,
                    self.__probability_buffer,

                    cp.float64(delta_r[0]) if dp else cp.float32(delta_r[0]),
                    cp.float64(delta_r[1]) if dp else cp.float32(delta_r[1]),
                    cp.float64(delta_r[2]) if dp else cp.float32(delta_r[2]),

                    cp.uint32(self.__bottom_voxel[0]),
                    cp.uint32(self.__bottom_voxel[1]),
                    cp.uint32(self.__bottom_voxel[2]),

                    cp.uint32(N[0]),
                    cp.uint32(N[1]),
                    cp.uint32(N[2]),
                )
            )
            self.__probability_evolution = np.append(
                arr=self.__probability_evolution, values=self.__probability_buffer[0]
            )

    def get_probability_evolution_with_name(self):
        return self.__probability_evolution, self.__name

    def clear(self):
        self.__probability_evolution = np.empty(shape=0, dtype=np.float64)

    def synchronize(self):
        self.__cuda_stream.synchronize()

class PlaneProbabilityCurrent:
    __name: str
    __center_bohr_radii_3: np.array
    __normal_vector_3: np.array
    __enable_image: np.array
    __kernel: cp.RawKernel
    __size_bohr_radii_2: np.array
    __resolution_2: np.array
    __probability_current_density: cp.ndarray
    __probability_current_buffer: cp.array
    __probability_current_evolution: np.array
    __grid_size: np.array
    __block_size: np.array
    __shared_memory_bytes: int
    __delta_t: float
    __cuda_stream: cp.cuda.Stream

    def __init__(self, config: Dict, sim_state: SimState):
        self.__name = try_read_param(config, "name", "Probability current", "measurement.plane_probability_currents")
        self.__center_bohr_radii_3 = np.array(
            try_read_param(config, "center_bohr_radii_3", "measurement.plane_probability_currents")
        )
        self.__normal_vector_3 = math_utils.normalize(np.array(
            try_read_param(config, "normal_vector_3", "measurement.plane_probability_currents")
        ))
        self.__enable_image = try_read_param(config, "enable_image", "measurement.plane_probability_currents")
        self.__size_bohr_radii_2 = np.array(
            try_read_param(config, "size_bohr_radii_2", [60.0, 60.0], "measurement.plane_probability_currents"))
        self.__resolution_2 = np.array(try_read_param(config, "resolution_2", [512, 512], "measurement.plane_probability_currents"))
        self.__resolution_2 += np.array([1, 1], dtype=self.__resolution_2.dtype)  # Add one because of the inclusive requirement of Simpson
        for i in range(2):
            # Correct even voxel count to odd. (Because of the Simpson integration scheme)
            if self.__resolution_2[i] % 2 == 0:
                self.__resolution_2[i] -= 1

        probability_current_density_kernel = (
            Path("sources/cuda_kernels/probability_current_density.cu").read_text().replace("PATH_TO_SOURCES",
                                                                                            os.path.abspath("sources"))
            .replace("T_WF_FLOAT",
                     "double" if sim_state.is_double_precision() else "float"))
        self.__kernel = cp.RawKernel(
            probability_current_density_kernel,
            "probability_current_density_kernel"
        )
        float_type = (cp.float64 if sim_state.is_double_precision() else cp.float32)
        self.__probability_current_density = cp.zeros(shape=[self.__resolution_2[0], self.__resolution_2[1]], dtype=float_type)
        self.__probability_current_buffer = cp.array([0.0], dtype=float_type)
        self.__probability_current_evolution = np.empty(shape=0, dtype=float_type)
        self.__delta_t = sim_state.get_delta_time_h_bar_per_hartree()
        self.__grid_size, self.__block_size = math_utils.get_grid_size_block_size(self.__resolution_2)
        self.__shared_memory_bytes = self.__block_size[0] * self.__block_size[1] * cp.dtype(float_type).itemsize
        print(f"Shared mem size = {self.__shared_memory_bytes} bytes")
        print(f"Block count = {self.__grid_size}")
        print(f"Thread count = {self.__block_size}")
        self.__cuda_stream = cp.cuda.Stream()

    def get_name(self):
        return self.__name

    def set_name(self, n: str):
        self.__name = n

    def is_enable_image(self):
        return self.__enable_image

    def calculate(self, sim_state: SimState):
        with self.__cuda_stream:
            self.__probability_current_buffer[0] = 0.0  # Clear buffer
            dp = sim_state.is_double_precision()
            self.__kernel(
                self.__grid_size,
                self.__block_size,
                (
                    sim_state.get_wave_function(),
                    self.__probability_current_density,
                    self.__probability_current_buffer,

                    cp.float64(sim_state.get_particle_mass()) if dp else cp.float32(sim_state.get_particle_mass()),

                    cp.float64(sim_state.get_delta_x_bohr_radii_3()[0]) if dp else cp.float32(sim_state.get_delta_x_bohr_radii_3()[0]),
                    cp.float64(sim_state.get_delta_x_bohr_radii_3()[1]) if dp else cp.float32(sim_state.get_delta_x_bohr_radii_3()[1]),
                    cp.float64(sim_state.get_delta_x_bohr_radii_3()[2]) if dp else cp.float32(sim_state.get_delta_x_bohr_radii_3()[2]),

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
                ),
                shared_mem=self.__shared_memory_bytes
            )

            self.__probability_current_evolution = (
                np.append(arr=self.__probability_current_evolution, values=self.__probability_current_buffer[0]))

    def get_probability_current_evolution_with_name(self):
        return self.__probability_current_evolution, self.__name

    def get_integrated_probability_current_evolution_with_name(self):
        return math_utils.indefinite_simpson_integral(
            self.__probability_current_evolution, self.__delta_t
        ), self.__name

    def synchronize(self):
        self.__cuda_stream.synchronize()

class ExpectedLocation:
    __enable_image: bool
    __expected_location_buffer: cp.array
    __expected_location_squared_buffer: cp.array
    __expected_location_evolution: np.array = np.zeros(shape=(0, 3), dtype=np.float64)
    __standard_deviation_evolution: np.array = np.zeros(shape=(0, 3), dtype=np.float64)
    __kernel: cp.RawKernel
    __kernel_grid_size: np.array
    __kernel_block_size: np.array
    __bottom_voxel: np.array
    __cuda_stream: cp.cuda.Stream

    def __init__(self, config: Dict, sim_state: SimState):
        self.__enable_image = try_read_param(config, "measurement.expected_location.enable_image", False)
        self.__bottom_voxel = sim_state.get_observation_box_bottom_corner_voxel_3()
        top_voxel = sim_state.get_observation_box_top_corner_voxel_3()
        top_voxel += np.array([1, 1, 1], dtype=top_voxel.dtype)  # Add because of the inclusive boundary requirement of the Simpson scheme
        sample_count = top_voxel - self.__bottom_voxel
        for i in range(3):
            # Offset boundary if even voxels are included. (Because of the Simpson integration scheme.)
            if sample_count[i] % 2 == 0:
                print(Fore.RED + f"Expected location: truncating integration area along {i}. axis!" + Style.RESET_ALL)
                sample_count[i] -= 1
                top_voxel[i] -= 1
        kernel_shape = (
            math_utils.nearest_power_of_2(sample_count[0]),
            math_utils.nearest_power_of_2(sample_count[1]),
            math_utils.nearest_power_of_2(sample_count[2]),
        )
        self.__kernel_grid_size, self.__kernel_block_size = math_utils.get_grid_size_block_size(kernel_shape, reduced_thread_count=True)
        kernel_source = (Path("sources/cuda_kernels/expected_location.cu").read_text()
                         .replace("PATH_TO_SOURCES", os.path.abspath("sources"))
                         .replace("T_WF_FLOAT", "double" if sim_state.is_double_precision() else "float"))

        func_name = f"expected_location_kernel<{sample_count[0]}, {sample_count[1]}, {sample_count[2]}>"
        self.__kernel = cp.RawModule(
            code=kernel_source,
            name_expressions=[func_name],
        ).get_function(func_name)
        self.__expected_location_buffer = cp.array([0.0, 0.0, 0.0], dtype=cp.float64 if sim_state.is_double_precision() else cp.float32)
        self.__expected_location_squared_buffer = cp.array([0.0, 0.0, 0.0], dtype=cp.float64 if sim_state.is_double_precision() else cp.float32)
        self.__cuda_stream = cp.cuda.Stream()

    def is_enable_image(self):
        return self.__enable_image

    def calculate(self, sim_state: SimState):
        with self.__cuda_stream:
            self.__expected_location_buffer[0] = 0.0
            self.__expected_location_buffer[1] = 0.0
            self.__expected_location_buffer[2] = 0.0

            self.__expected_location_squared_buffer[0] = 0.0
            self.__expected_location_squared_buffer[1] = 0.0
            self.__expected_location_squared_buffer[2] = 0.0

            wave_function = sim_state.get_wave_function()
            delta_r = sim_state.get_delta_x_bohr_radii_3()
            N = sim_state.get_number_of_voxels_3()
            dp = sim_state.is_double_precision()
            self.__kernel(
                self.__kernel_grid_size,
                self.__kernel_block_size,
                (
                    wave_function,
                    self.__expected_location_buffer,
                    self.__expected_location_squared_buffer,

                    (cp.float64(delta_r[0]) if dp else cp.float32(delta_r[0])),
                    (cp.float64(delta_r[1]) if dp else cp.float32(delta_r[1])),
                    (cp.float64(delta_r[2]) if dp else cp.float32(delta_r[2])),

                    cp.int32(self.__bottom_voxel[0]),
                    cp.int32(self.__bottom_voxel[1]),
                    cp.int32(self.__bottom_voxel[2]),

                    cp.int32(N[0]),
                    cp.int32(N[1]),
                    cp.int32(N[2])
                )
            )

            e_r = cp.asnumpy(self.__expected_location_buffer).reshape((1, 3))
            e_r_2 = cp.asnumpy(self.__expected_location_squared_buffer).reshape((1, 3))
            self.__expected_location_evolution = np.concatenate(
                (self.__expected_location_evolution, e_r),
                axis=0
            )
            self.__standard_deviation_evolution = np.concatenate(
                (
                    self.__standard_deviation_evolution,
                    np.sqrt(e_r_2 - np.power(e_r, 2))
                ),
                axis=0
            )

    def synchronize(self):
        self.__cuda_stream.synchronize()

    def get_expected_location_evolution(self):
        return self.__expected_location_evolution

    def get_standard_deviation_evolution(self):
        return self.__standard_deviation_evolution

class ProjectedMeasurement:
    probability_density: np.array
    min_voxel: int
    max_voxel: int
    near_voxel: int  # in directions of summing axes
    far_voxel: int  # in directions of summing axes
    left_edge_bohr_radii: float
    right_edge_boh_radii: float
    sum_axis: tuple
    label: str
    scale_factor: float = 1.0
    offset: float = 0.0

    def __init__(self,
                 min_voxel: int,
                 max_voxel: int,
                 left_edge: float,
                 right_edge: float,
                 sum_axis: tuple,
                 label: str,
                 near_voxel: int = None,
                 far_voxel: int = None,
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
            self.near_voxel = self.min_voxel  # let's assume cube shaped viewing window
        else:
            self.near_voxel = near_voxel
        if far_voxel is None:
            self.far_voxel = self.max_voxel  # let's assume cube shaped viewing window
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
    __expected_location: ExpectedLocation = None

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
        self.__volumetric_animation_capture_interval = try_read_param(config,
                                                                      "measurement.volumetric.animation_frame_capture_iteration_interval",
                                                                      5)
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

        # Volume probabilities:
        volume_confs = try_read_param(config, "measurement.volume_probabilities", [])
        for conf in volume_confs:
            new_v = VolumeProbability(conf, sim_state)
            self._resolve_naming_conflicts(self.__volume_probabilities, new_v)
            self.__volume_probabilities.append(new_v)

        # Plane probability currents:
        plane_confs = try_read_param(config, "measurement.plane_probability_currents", [])
        for conf in plane_confs:
            new_p = PlaneProbabilityCurrent(conf, sim_state)
            self._resolve_naming_conflicts(self.__plane_probability_currents, new_p)
            self.__plane_probability_currents.append(new_p)

        # Expected location:
        if try_read_param(config, "measurement.expected_location.enable_image", False):
            self.__expected_location = ExpectedLocation(config, sim_state)

    def write_wave_function_to_file(self, sim_state: SimState, iter_data: IterData):
        if iter_data.i % sim_state.get_wave_function_save_interval() == 0:
            if not os.path.exists(os.path.join(sim_state.get_output_dir(), f"wave_function")):
                os.makedirs(os.path.join(sim_state.get_output_dir(), f"wave_function"), exist_ok=True)
            try:
                cp.save(arr=sim_state.get_view_into_wave_function(),
                        file=os.path.join(sim_state.get_output_dir(), f"wave_function/wave_function_{iter_data.i:04d}.npy"))
            except IOError:
                print(Fore.RED + "\nERROR: Failed writing file: " + os.path.join(sim_state.get_output_dir(),
                                                                                 f"wave_function/wave_function_{iter_data.i:04d}.npy") + Style.RESET_ALL)

    def measure_and_render(self, sim_state: SimState, iter_data: IterData):
        # Save wave function:
        if sim_state.is_wave_function_saving():
            self.write_wave_function_to_file(sim_state=sim_state, iter_data=iter_data)

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
                volume.calculate(sim_state)

        for plane in self.__plane_probability_currents:
            if plane.is_enable_image():
                plane.calculate(sim_state)

        if self.__expected_location is not None:
            self.__expected_location.calculate(sim_state)

        # Synchronize separate CUDA streams:
        for volume in self.__volume_probabilities:
            if volume.is_enable_image():
                volume.synchronize()

        for plane in self.__plane_probability_currents:
            if plane.is_enable_image():
                plane.synchronize()

        if self.__expected_location is not None:
            self.__expected_location.synchronize()


    def finish(self, sim_state: SimState):
        # Animations:
        if self.__enable_volumetric_animation:
            self.__animation_writer_volumetric.finish()
        if self.__enable_per_axis_animation:
            self.__animation_writer_per_axis.finish()

        # Expected location:
        if self.__expected_location is not None:
            expected_location_evolution = self.__expected_location.get_expected_location_evolution()
            expected_location_evolution_with_label = list(zip(expected_location_evolution.T, ["X axis", "Y axis", "Z axis"]))
            np.save(os.path.join(sim_state.get_output_dir(), f"expected_location_evolution.npy"), expected_location_evolution)
            plot.plot_probability_evolution(
                out_dir=sim_state.get_output_dir(),
                file_name="expected_location_evolution.png",
                title="Expected location evolution",
                y_label="Expected location [Bohr radius]",
                probability_evolutions=expected_location_evolution_with_label,
                delta_t=sim_state.get_delta_time_h_bar_per_hartree(),
                show_fig=self.__show_figures,
                y_min=np.min(sim_state.get_observation_box_bottom_corner_bohr_radii_3()),
                y_max=np.max(sim_state.get_observation_box_top_corner_bohr_radii_3()),
            )

        # Standard deviation:
        if self.__expected_location is not None:
            standard_deviation_evolution = self.__expected_location.get_standard_deviation_evolution()
            standard_deviation_evolution_with_label = list(zip(standard_deviation_evolution.T, ["X axis", "Y axis", "Z axis"]))
            np.save(os.path.join(sim_state.get_output_dir(), f"standard_deviation_evolution.npy"), standard_deviation_evolution)
            plot.plot_probability_evolution(
                out_dir=sim_state.get_output_dir(),
                file_name="standard_deviation_evolution.png",
                title="Standard deviation evolution",
                y_label="Standard deviation [Bohr radius]",
                probability_evolutions=standard_deviation_evolution_with_label,
                delta_t=sim_state.get_delta_time_h_bar_per_hartree(),
                show_fig=self.__show_figures,
                y_min=0.0,
                y_max=np.max(sim_state.get_observation_box_top_corner_bohr_radii_3()) * 0.5,
            )

        # Volume probabilities:
        volume_probability_evolutions = []
        for v in self.__volume_probabilities:
            if v.is_enable_image():
                prob_with_name = v.get_probability_evolution_with_name()
                volume_probability_evolutions.append(prob_with_name)
                np.save(os.path.join(sim_state.get_output_dir(), f"volume_probability_evolution_{prob_with_name[1]}.npy"), prob_with_name[0])
                dwell_time = math_utils.indefinite_simpson_integral(prob_with_name[0], sim_state.get_delta_time_h_bar_per_hartree())
                print(f"Dwell time in {prob_with_name[1]}: {dwell_time[-1]:.5f} ħ/Hartree")
        if len(volume_probability_evolutions) > 0:
            sum = np.array(
                np.zeros(shape=volume_probability_evolutions[0][0].shape, dtype=volume_probability_evolutions[0][0].dtype).tolist()
            )
            for evolution in volume_probability_evolutions:
                sum = np.add(sum, np.array(evolution[0].tolist()))
            volume_probability_evolutions.append([sum, "Sum"])
            plot.plot_probability_evolution(
                out_dir=sim_state.get_output_dir(),
                file_name="volume_probability_evolution.png",
                title="Volume probability evolution",
                y_label="Probability",
                probability_evolutions=volume_probability_evolutions,
                delta_t=sim_state.get_delta_time_h_bar_per_hartree(),
                show_fig=self.__show_figures,
                y_min=-0.05,
                y_max=1.05
            )

        # Plane probability currents:
        probability_current_evolutions = []
        for pc in self.__plane_probability_currents:
            if pc.is_enable_image():
                pc_with_name = pc.get_probability_current_evolution_with_name()
                probability_current_evolutions.append(pc_with_name)
                np.save(os.path.join(sim_state.get_output_dir(), f"probability_current_evolution_{pc_with_name[1]}.npy"), pc_with_name[0])
        if len(probability_current_evolutions) > 0:
            plot.plot_probability_evolution(
                out_dir=sim_state.get_output_dir(),
                file_name="probability_current_evolution.png",
                title="Probability current evolution",
                y_label="Probability current",
                probability_evolutions=probability_current_evolutions,
                delta_t=sim_state.get_delta_time_h_bar_per_hartree(),
                show_fig=self.__show_figures,
                y_min=-1.1,
                y_max=1.1
            )

        # Integrated probability current:
        integrated_probability_current_evolutions = []
        for pc in self.__plane_probability_currents:
            if pc.is_enable_image():
                ipc_with_name = pc.get_integrated_probability_current_evolution_with_name()
                integrated_probability_current_evolutions.append(ipc_with_name)
                np.save(os.path.join(sim_state.get_output_dir(), f"integrated_probability_current_{ipc_with_name[1]}.npy"), ipc_with_name[0])
                print(f"Transfer probability on {ipc_with_name[1]}: {ipc_with_name[0][-1]:.5f}")    # Print last element
        if len(integrated_probability_current_evolutions) > 0:
            plot.plot_probability_evolution(
                out_dir=sim_state.get_output_dir(),
                file_name="integrated_probability_current_evolution.png",
                title="Integrated probability current evolution",
                y_label="Probability",
                probability_evolutions=integrated_probability_current_evolutions,
                delta_t=sim_state.get_delta_time_h_bar_per_hartree(),
                show_fig=self.__show_figures,
                y_min=-1.1,
                y_max=1.1
            )
