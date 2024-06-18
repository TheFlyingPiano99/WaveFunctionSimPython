import cupy as cp
import numpy as np
import sources.math_utils as math_utils
import sources.operators
from sources import potential
from enum import Enum
from typing import Dict
from sources.config_read_helper import try_read_param
from sources import wave_packet, potential, operators
from colorama import Fore, Style
import os
import io
from pathlib import Path


class SimulationMethod(Enum):
    FOURIER = 0
    POWER_SERIES = 1


class SimState:
    __wave_packet: wave_packet.GaussianWavePacket
    __number_of_voxels_3 = np.array([128, 128, 128])
    __observation_box_bottom_corner_voxel_3: np.array
    __observation_box_top_corner_voxel_3: np.array
    __observation_box_bottom_corner_bohr_radii_3: np.array
    __observation_box_top_corner_bohr_radii_3: np.array
    __de_broglie_wave_length_bohr_radii: float
    __simulated_volume_dimensions_bohr_radii_3: np.array([120.0, 120.0, 120.0])
    __delta_x_bohr_radii_3: np.array([0.0, 0.0, 0.0])
    __upper_limit_on_delta_time_h_per_hartree: float
    __delta_time_h_bar_per_hartree: float
    __wave_tensor: cp.ndarray
    __kinetic_operator: cp.ndarray
    __potential_operator: cp.ndarray
    __probability_density: cp.ndarray
    __localised_potential_hartree: cp.ndarray
    __localised_potential_to_visualize_hartree: cp.ndarray
    __coulomb_potential: cp.ndarray
    __output_dir: str = ""
    __cache_dir: str = ""
    __use_cache: bool = True
    __enable_visual_output: bool = True
    __double_precision_calculation: bool = False
    __enable_wave_function_saving: bool = True
    __wave_function_save_iteration_interval: int = 1
    __absorbing_boundary_condition: potential.AbsorbingBoundaryCondition = None
    __potential_walls: list[potential.PotentialWall] = []
    __is_dynamic_potential_mode: bool = False
    __simulation_method: SimulationMethod = SimulationMethod.FOURIER
    __pre_initialized_potential: potential.PreInitializedPotential
    __order_of_approximation: int = 10
    __pingpong_buffer: list[cp.ndarray] = []
    __next_s_kernel: cp.RawKernel = None
    __kinetic_operator_kernel: cp.RawKernel = None
    __potential_operator_kernel: cp.RawKernel = None
    __kernel_grid_size = (32, 32, 32)
    __kernel_block_size = (32, 32, 32)
    __wave_numbers: list[cp.array]  # Used to simplify the calculation of wave number coordinates in k space

    def __init__(self, config: Dict):

        # Set paths:
        self.__cache_dir = try_read_param(config, "paths.cache_dir", "./cache")
        self.__output_dir = try_read_param(config, "paths.output_dir", "./output")

        # Set simulation method:
        method_name = try_read_param(config, "simulation.method", "fourier")
        method_name = method_name.lower()   # Convert to lower-case
        if method_name not in ["fft", "fourier", "fdtd", "power_series"]:
            self.__simulation_method = SimulationMethod.FOURIER
        else:
            self.__simulation_method = SimulationMethod.FOURIER if (
                    method_name in ["fft", "fourier"]
            ) else SimulationMethod.POWER_SERIES    # Only two options currently

        self.__order_of_approximation = try_read_param(config, "simulation.order_of_approximation", 10)
        self.__is_dynamic_potential_mode = try_read_param(config, "simulation.is_dynamic_potential_mode", True)
        self.__enable_wave_function_saving = try_read_param(config, "simulation.enable_wave_function_saving", False)
        self.__wave_function_save_iteration_interval = try_read_param(config, "simulation.wave_function_save_iteration_interval", 1)
        self.__double_precision_calculation = try_read_param(config, "simulation.double_precision_calculation", False)

        # Wave packet:
        self.__wave_packet = wave_packet.GaussianWavePacket(config)

        # Set Volume parameters:
        self.__simulated_volume_dimensions_bohr_radii_3 = np.array(
            try_read_param(config, "volume.simulated_volume_dimensions_bohr_radii_3", [100.0, 100.0, 100.0])
        )
        self.__number_of_voxels_3 = try_read_param(config, "volume.number_of_voxels_3", [256, 256, 256])
        self.__coulomb_potential = cp.zeros(shape=self.__number_of_voxels_3)    # Unused in current version.
        self.__delta_x_bohr_radii_3 = self.__simulated_volume_dimensions_bohr_radii_3 / self.__number_of_voxels_3
        self.__upper_limit_on_delta_time_h_per_hartree = (
            4.0
            / np.pi
            * (3.0 * np.max(self.__delta_x_bohr_radii_3) * np.max(self.__delta_x_bohr_radii_3))
            / 3.0
        )  # Based on reasoning from the Web-Schrödinger paper
        self.__delta_time_h_bar_per_hartree = config["simulation"][
            "delta_time_h_bar_per_hartree"
        ]

        # Set observation box boundaries:
        self.__observation_box_bottom_corner_bohr_radii_3 = np.array(
            try_read_param(config, "volume.observation_box_bottom_corner_bohr_radii_3", [-25.0, -25.0, -25.0])
        )
        self.__observation_box_top_corner_bohr_radii_3 = np.array(
            try_read_param(config, "volume.observation_box_top_corner_bohr_radii_3", [25.0, 25.0, 25.0])
        )
        # Flip coordinates if inverted:
        for i in range(3):
            if (
                    self.__observation_box_bottom_corner_bohr_radii_3[i] >
                    self.__observation_box_top_corner_bohr_radii_3[i]
            ):
                temp = self.__observation_box_bottom_corner_bohr_radii_3[i]
                self.__observation_box_bottom_corner_bohr_radii_3[i] = self.__observation_box_top_corner_bohr_radii_3[i]
                self.__observation_box_top_corner_bohr_radii_3[i] = temp

        # Clip boundaries if out of simulated volume:
        for i in range(3):
            if (
                    self.__observation_box_bottom_corner_bohr_radii_3[i]
                    < -self.__simulated_volume_dimensions_bohr_radii_3[i] * 0.5
            ):
                self.__observation_box_bottom_corner_bohr_radii_3[i] = -self.__simulated_volume_dimensions_bohr_radii_3[i] * 0.5
            if (
                    self.__observation_box_top_corner_bohr_radii_3[i]
                    > self.__simulated_volume_dimensions_bohr_radii_3[i] * 0.5
            ):
                self.__observation_box_top_corner_bohr_radii_3[i] = self.__simulated_volume_dimensions_bohr_radii_3[i] * 0.5

        self.__observation_box_bottom_corner_voxel_3 = np.array(
            math_utils.transform_center_origin_to_corner_origin_system(
                self.__observation_box_bottom_corner_bohr_radii_3,
                self.__simulated_volume_dimensions_bohr_radii_3
            )
            / self.__delta_x_bohr_radii_3,
            dtype=np.int32,
        )
        self.__observation_box_top_corner_voxel_3 = np.array(
            math_utils.transform_center_origin_to_corner_origin_system(
                self.__observation_box_top_corner_bohr_radii_3,
                self.__simulated_volume_dimensions_bohr_radii_3,
            )
            / self.__delta_x_bohr_radii_3,
            dtype=np.int32,
        )

        # Correct top voxel if out of simulated voxel count:
        for i in range(3):
            if (self.__observation_box_top_corner_voxel_3[i] >= self.__number_of_voxels_3[i]):
                self.__observation_box_top_corner_voxel_3[i] = self.__number_of_voxels_3[i] - 1
        # Correct bottom voxel if out of simulated voxel count:
        for i in range(3):
            if (self.__observation_box_bottom_corner_voxel_3[i] < 0):
                self.__observation_box_bottom_corner_voxel_3[i] = 0
        print(f"Observation box bottom corner voxel: ({self.__observation_box_bottom_corner_voxel_3[0]}, {self.__observation_box_bottom_corner_voxel_3[1]}, {self.__observation_box_bottom_corner_voxel_3[2]})")
        print(f"Observation box top corner voxel: ({self.__observation_box_top_corner_voxel_3[0]}, {self.__observation_box_top_corner_voxel_3[1]}, {self.__observation_box_top_corner_voxel_3[2]})")

        # Load potential descriptions:
        # Set pre initialized potential path (can be empty string):
        self.__pre_initialized_potential = potential.PreInitializedPotential(config)

        if (len(self.__pre_initialized_potential.path) == 0) or not os.path.exists(self.__pre_initialized_potential.path):
            self.__pre_initialized_potential.enable = False     # Force disable if unavailable

        # Init absorbing potential:
        self.__absorbing_boundary_condition = potential.AbsorbingBoundaryCondition(
            config,
            bottom_corner=self.__observation_box_bottom_corner_bohr_radii_3,
            top_corner=self.__observation_box_top_corner_bohr_radii_3
        )  # It processes the viewing boundaries
        # Config potential walls:
        walls = try_read_param(config, "potential.walls", [])
        for w in walls:
            self.__potential_walls.append(potential.PotentialWall(w))

    def initialize_state(self):
        print(self.get_sim_state_description_text(use_colors=True))

        print(
            "\n***************************************************************************************\n"
        )

        print(Fore.GREEN + "Initializing wave packet" + Style.RESET_ALL)

        full_init = True
        if self.__use_cache:
            try:
                self.__wave_tensor = cp.load(file=os.path.join(self.__cache_dir, "gaussian_wave_packet.npy"))
                full_init = False
            except OSError:
                print("No cached gaussian_wave_packet.npy found.")

        if full_init:
            self.__wave_tensor = cp.asarray(
                self.__wave_packet.init_wave_packet(
                    self.__delta_x_bohr_radii_3,
                    self.__number_of_voxels_3,
                    self.__double_precision_calculation
                )
            )
            #cp.save(file=os.path.join(self.__cache_dir, "gaussian_wave_packet.npy"), arr=self.__wave_tensor)
        # Normalize:
        self.__probability_density = math_utils.square_of_abs(self.__wave_tensor)
        dxdydz = self.__delta_x_bohr_radii_3[0] * self.__delta_x_bohr_radii_3[1] * self.__delta_x_bohr_radii_3[2]
        sum_probability = cp.sum(self.__probability_density) * dxdydz
        print(f"Sum of probabilities = {sum_probability:.8f}")
        self.__wave_tensor = self.__wave_tensor / (sum_probability ** 0.5)
        self.__probability_density = math_utils.square_of_abs(self.__wave_tensor)
        sum_probability = cp.sum(self.__probability_density) * dxdydz
        print(f"Sum of probabilities after normalization = {sum_probability:.8f}")

        full_init = True
        if self.__use_cache:
            try:
                self.__localised_potential_hartree = cp.asarray(
                    np.load(file=os.path.join(self.__cache_dir, "localized_potential.npy")))
                self.__localised_potential_to_visualize_hartree = cp.asarray(
                    np.load(file=os.path.join(self.__cache_dir, "localized_potential_to_visualize.npy")))
                full_init = False
            except OSError:
                print("No cached localized_potential.npy found.")

        if full_init:
            self.__localised_potential_hartree = cp.zeros(
                shape=self.__number_of_voxels_3, dtype=cp.complex64
            )
            self.__localised_potential_to_visualize_hartree = cp.zeros(
                shape=self.__number_of_voxels_3, dtype=cp.complex64
            )

            if self.__pre_initialized_potential.enable:
                    print("Loading pre-initialized potential")
                    try:
                        pre_init_pot = cp.asarray(np.load(file=self.__pre_initialized_potential.path))
                        if pre_init_pot.shape == self.__localised_potential_hartree.shape:
                            self.__localised_potential_hartree += pre_init_pot
                            if self.__pre_initialized_potential.visible:
                                self.__localised_potential_to_visualize_hartree += pre_init_pot
                        else:
                            print(Fore.RED + "Pre-initialized potential has the wrong tensor shape!" + Style.RESET_ALL)
                    except IOError:
                        print(Fore.RED + "Found pre-initialized potential but failed to load!" + Style.RESET_ALL)

            print("Creating draining potential.")
            self.__localised_potential_hartree = self.__absorbing_boundary_condition.add_potential(
                V=self.__localised_potential_hartree,
                delta_x_3=self.__delta_x_bohr_radii_3,
            )

            for wall in self.__potential_walls:
                print("Creating wall.")
                tensor = wall.add_potential(
                    delta_x_3=self.__delta_x_bohr_radii_3,
                    V=cp.zeros(shape=self.__number_of_voxels_3, dtype=cp.complex64),
                )
                self.__localised_potential_hartree += tensor
                if wall.visible:
                    self.__localised_potential_to_visualize_hartree += tensor


            np.save(
                file=os.path.join(self.__cache_dir, "localized_potential.npy"),
                arr=cp.asnumpy(self.__localised_potential_hartree),
            )
            np.save(file=os.path.join(self.__cache_dir, "localized_potential_to_visualize.npy"),
                    arr=cp.asnumpy(self.__localised_potential_to_visualize_hartree))

        shape = self.__number_of_voxels_3
        self.__kernel_grid_size = math_utils.get_grid_size(shape)
        self.__kernel_block_size = (shape[0] // self.__kernel_grid_size[0], shape[1] // self.__kernel_grid_size[1], shape[2] // self.__kernel_grid_size[2])
        if self.__simulation_method == SimulationMethod.FOURIER:
            kinetic_operator_kernel_source = (Path("sources/cuda_kernels/kinetic_operator.cu")
                                              .read_text().replace("PATH_TO_SOURCES", os.path.abspath("sources"))
                                              .replace("T_WF_FLOAT",
                                                       "double" if self.__double_precision_calculation else "float"))

            self.__kinetic_operator_kernel = cp.RawKernel(kinetic_operator_kernel_source,
                                                   'kinetic_operator_kernel',
                                                   enable_cooperative_groups=False)
            potential_operator_kernel_source = (Path("sources/cuda_kernels/potential_operator.cu")
                                                .read_text().replace("PATH_TO_SOURCES", os.path.abspath("sources"))
                                                .replace("T_WF_FLOAT",
                                                         "double" if self.__double_precision_calculation else "float"))
            self.__potential_operator_kernel = cp.RawKernel(potential_operator_kernel_source,
                                                   'potential_operator_kernel',
                                                   enable_cooperative_groups=False)
            self.__wave_numbers = []
            self.__wave_numbers.append(cp.fft.fftfreq(n=shape[0], d=self.__delta_x_bohr_radii_3[0]))
            self.__wave_numbers.append(cp.fft.fftfreq(n=shape[1], d=self.__delta_x_bohr_radii_3[1]))
            self.__wave_numbers.append(cp.fft.fftfreq(n=shape[2], d=self.__delta_x_bohr_radii_3[2]))

        elif self.__simulation_method == SimulationMethod.POWER_SERIES:
            # Define the kernel for the power series method
            next_s_kernel_source = (Path("sources/cuda_kernels/power_series_operator.cu").read_text().replace(
                "PATH_TO_SOURCES", os.path.abspath("sources"))
                .replace("T_WF_FLOAT",
                         "double" if self.__double_precision_calculation else "float"))
            self.__next_s_kernel = cp.RawKernel(
                next_s_kernel_source,
                "next_s",
                enable_cooperative_groups=False
            )
            self.__pingpong_buffer = [
                cp.zeros(shape=self.__wave_tensor.shape, dtype=self.__wave_tensor.dtype),
                cp.zeros(shape=self.__wave_tensor.shape, dtype=self.__wave_tensor.dtype)
            ]  # s is used as a pair of pingpong buffers to store power series elements

    def get_delta_time_h_bar_per_hartree(self):
        return self.__delta_time_h_bar_per_hartree

    def get_delta_x_bohr_radii_3(self):
        return self.__delta_x_bohr_radii_3

    def set_wave_function(self, wave_func: cp.ndarray):
        self.__wave_tensor = wave_func

    def get_wave_function(self):
        return self.__wave_tensor

    def is_wave_function_saving(self):
        return self.__enable_wave_function_saving

    def get_wave_function_save_interval(self):
        return self.__wave_function_save_iteration_interval

    def get_simulation_method(self):
        return self.__simulation_method

    def is_use_cache(self):
        return self.__use_cache

    def set_use_cache(self, uc: bool):
        self.__use_cache = uc

    def get_cache_dir(self):
        return self.__cache_dir

    def get_output_dir(self):
        return self.__output_dir

    def get_number_of_voxels_3(self):
        return self.__number_of_voxels_3

    def get_observation_box_bottom_corner_voxel_3(self):
        return self.__observation_box_bottom_corner_voxel_3

    def get_observation_box_top_corner_voxel_3(self):
        return self.__observation_box_top_corner_voxel_3

    def get_observation_box_bottom_corner_bohr_radii_3(self):
        return self.__observation_box_bottom_corner_bohr_radii_3

    def get_observation_box_top_corner_bohr_radii_3(self):
        return self.__observation_box_top_corner_bohr_radii_3

    def get_view_into_wave_function(self):
        return math_utils.cut_bounding_box(
            arr=self.__wave_tensor,
            bottom=self.__observation_box_bottom_corner_voxel_3,
            top=self.__observation_box_top_corner_voxel_3,
        )

    def is_double_precision_calculation(self):
        return self.__double_precision_calculation

    def _transform_physical_coordinate_to_voxel_3(self, pos_bohr_radii_3: np.array):
        return math_utils.transform_center_origin_to_corner_origin_system(
            pos_bohr_radii_3,
            self.__simulated_volume_dimensions_bohr_radii_3
        ) / self.__delta_x_bohr_radii_3

    def _transform_voxel_to_physical_coordinate_3(self, voxel_3: np.array):
        return math_utils.transform_corner_origin_to_center_origin_system(
            voxel_3,
            self.__number_of_voxels_3
        ) * self.__delta_x_bohr_radii_3

    def get_view_into_probability_density(
            self,
            bottom_corner_bohr_radii: np.array = None,
            top_corner_bohr_radii: np.array = None
    ):
        if not (bottom_corner_bohr_radii is None) and not (top_corner_bohr_radii is None):
            bottom_voxel_3 = self._transform_physical_coordinate_to_voxel_3(bottom_corner_bohr_radii)
            top_voxel_3 = self._transform_physical_coordinate_to_voxel_3(top_corner_bohr_radii)
            return math_utils.cut_bounding_box(
                arr=self.__probability_density,
                bottom=bottom_voxel_3,
                top=top_voxel_3
            )
        return math_utils.cut_bounding_box(
            arr=self.__probability_density,
            bottom=self.__observation_box_bottom_corner_voxel_3,
            top=self.__observation_box_top_corner_voxel_3,
        )

    def get_view_into_potential(self):
        return math_utils.cut_bounding_box(
            arr=cp.real(self.__localised_potential_to_visualize_hartree),
            bottom=self.__observation_box_bottom_corner_voxel_3,
            top=self.__observation_box_top_corner_voxel_3,
        )

    def get_view_into_complex_potential(self):
        return math_utils.cut_bounding_box(
            arr=self.__localised_potential_to_visualize_hartree,
            bottom=self.__observation_box_bottom_corner_voxel_3,
            top=self.__observation_box_top_corner_voxel_3,
        )

    def get_view_into_coulomb_potential(self):
        return math_utils.cut_bounding_box(
            arr=self.__coulomb_potential,
            bottom=self.__observation_box_bottom_corner_voxel_3,
            top=self.__observation_box_top_corner_voxel_3,
        )

    def get_copy_of_wave_function(self):
        return cp.copy(self.__wave_tensor)

    def get_particle_mass(self):
        return self.__wave_packet.get_particle_mass_electron_rest_mass()

    def update_probability_density(self):
        self.__probability_density = math_utils.square_of_abs(
            self.__wave_tensor
        )

    def update_potential(self):
        if not self.__is_dynamic_potential_mode:
            return
        for w in self.__potential_walls:
            # Advect:
            w.center_bohr_radii_3 = w.center_bohr_radii_3 + w.velocity_bohr_radius_hartree_per_h_bar_3 * self.__delta_time_h_bar_per_hartree
            if w.center_bohr_radii_3[0] < 0.0:  # Stop at zero (For testing only)
                w.center_bohr_radii_3[0] = 0.0

            """
            # Rotate around Z axis:
            w.normal_bohr_radii_3 = np.dot(
                math_utils.rotation_matrix(
                    np.array([0, 0, 1]),
                    w.angular_velocity_rad_hartree_per_h_bar[2] * self.delta_time_h_bar_per_hartree
                ),
                w.normal_bohr_radii_3
            )
            # Rotate around Y axis:
            w.normal_bohr_radii_3 = np.dot(
                math_utils.rotation_matrix(
                    np.array([0, 1, 0]),
                    w.angular_velocity_rad_hartree_per_h_bar[1] * self.delta_time_h_bar_per_hartree
                ),
                w.normal_bohr_radii_3
            )
            # Rotate around X axis:
            w.normal_bohr_radii_3 = np.dot(
                math_utils.rotation_matrix(
                    np.array([1, 0, 0]),
                    w.angular_velocity_rad_hartree_per_h_bar[0] * self.delta_time_h_bar_per_hartree
                ),
                w.normal_bohr_radii_3
            )
            w.potential_hartree = w.potential_change_rate_hartree_2_per_h_bar * self.delta_time_h_bar_per_hartree
        """

        self.__localised_potential_hartree, self.__localised_potential_to_visualize_hartree = potential.generate_potential_from_walls_and_drain(
            V=self.__localised_potential_hartree,
            V_vis=self.__localised_potential_to_visualize_hartree,
            delta_x_3=self.__delta_x_bohr_radii_3,
            absorbing_pontetial=self.__absorbing_boundary_condition,
            walls=self.__potential_walls
        )

    def write_potential_wall_warnings(self, text, wall_potential, wall, use_colors):
        time_times_potential = wall_potential * self.__delta_time_h_bar_per_hartree
        text.write(f"Obstacle wall potential is {wall_potential} Hartree.\n")
        if abs((time_times_potential / cp.pi) - int(time_times_potential / cp.pi)) < 0.05:
            text.write(
                (Fore.RED if use_colors else "")
                + f"WARNING: delta_t * wall_max_potential too close to multiply of pi! ({time_times_potential})\n"
                + (Style.RESET_ALL if use_colors else "")
            )
        thickness = wall["thickness_bohr_radii"]
        text.write(f"Wall thickness is {thickness} Bohr radii.\n")
        if thickness < self.__de_broglie_wave_length_bohr_radii:
            text.write(
                "This is thinner than the de Broglie wavelength of the particle.\n"
            )
        else:
            text.write(
                "This is thicker than the de Broglie wavelength of the particle.\n"
            )
        if time_times_potential > cp.pi:
            text.write(
                (Fore.RED if use_colors else "")
                + f"WARNING: delta_t * wall_max_potential = {time_times_potential} exceeds  pi!\n"
                + (Style.RESET_ALL if use_colors else "")
            )

    def get_potential_description_text(self, use_colors=False):
        text = io.StringIO()
        text.write(
            (Fore.GREEN if use_colors else "")
            + "Localised potential energy details:\n"
            + (Style.RESET_ALL if use_colors else "")
        )

        return text.getvalue()


    def get_sim_state_description_text(self, use_colors=False):
        text = io.StringIO()
        text.write(
            (Fore.GREEN if use_colors else "")
            + "Simulated system state:\n"
            + (Style.RESET_ALL if use_colors else "")
        )
        velocity_magnitude = (
            cp.dot(
                self.__wave_packet.get_initial_wp_velocity_bohr_radii_hartree_per_h_bar_3(),
                self.__wave_packet.get_initial_wp_velocity_bohr_radii_hartree_per_h_bar_3(),
            )
            ** 0.5
        )
        text.write(
            f"Mass of the particle is {self.__wave_packet.get_particle_mass_electron_rest_mass()} electron rest mass.\n"
            f"Initial velocity of the particle is {velocity_magnitude} Bohr radius Hartree / h-bar.\n"
        )

        momentum_magnitude = (
            cp.dot(
                self.__wave_packet.get_initial_wp_momentum_h_per_bohr_radii_3(),
                self.__wave_packet.get_initial_wp_momentum_h_per_bohr_radii_3(),
            )
            ** 0.5
        )
        text.write(
            f"Initial mean momentum of particle is {momentum_magnitude} h-bar / Bohr radius.\n"
        )
        text.write(
            f"De Broglie wavelength associated with the particle is {self.__wave_packet.get_de_broglie_wave_length_bohr_radii():.4f} Bohr radii.\n"
        )

        initial_kinetic_energy_hartree = (
            momentum_magnitude**2 / 2 / self.__wave_packet.get_particle_mass_electron_rest_mass()
        )
        text.write(
            f"Initial mean kinetic energy of the particle is {initial_kinetic_energy_hartree} Hartree.\n"
        )

        text.write(
            f"Dimensions of simulated volume is ({self.__simulated_volume_dimensions_bohr_radii_3[0]}, {self.__simulated_volume_dimensions_bohr_radii_3[1]}, {self.__simulated_volume_dimensions_bohr_radii_3[2]}) Bohr radii.\n"
        )

        text.write(f"Number of samples per axis is ({self.__number_of_voxels_3[0]}, {self.__number_of_voxels_3[1]}, {self.__number_of_voxels_3[2]}).\n")

        # Space resolution
        text.write(
            f"Space resolution is delta_r = ({self.__delta_x_bohr_radii_3[0]}, {self.__delta_x_bohr_radii_3[1]}, {self.__delta_x_bohr_radii_3[2]}) Bohr radii.\n"
        )
        if (
            np.max(self.__delta_x_bohr_radii_3)
            >= self.__wave_packet.get_de_broglie_wave_length_bohr_radii() / 2.0
        ):
            text.write(
                (Fore.RED if use_colors else "")
                + f"WARNING: max delta_x = {np.max(self.__delta_x_bohr_radii_3)} exceeds half of de Broglie wavelength!\n"
                + (Style.RESET_ALL if use_colors else "")
            )

        # The maximum allowed delta_time
        text.write(
            f"The maximal viable time resolution < {self.__upper_limit_on_delta_time_h_per_hartree} h-bar / Hartree\n"
        )

        # Time increment of simulation
        text.write(
            f"Time resolution is delta = {self.__delta_time_h_bar_per_hartree} h-bar / Hartree.\n"
        )
        if (
            self.__delta_time_h_bar_per_hartree
            > 0.5 * self.__upper_limit_on_delta_time_h_per_hartree
        ):
            text.write(
                (Fore.RED if use_colors else "")
                + "WARNING: delta_time exceeds theoretical limit!\n"
                + (Style.RESET_ALL if use_colors else "")
            )
        # Video
        if self.__enable_visual_output:
            pass

        return text.getvalue()

    def get_simulation_method_text(self, use_colors=False):
        text = io.StringIO()
        if self.__simulation_method == "fft":
            text.write((Fore.BLUE if use_colors else "") + "Using the Split-Operator Fourier method to simulate the time development.\n" + (Style.RESET_ALL if use_colors else ""))
        elif self.__simulation_method == "power_series":
            text.write((Fore.BLUE if use_colors else "") + "Using the Power Series method to simulate the time development.\n")
            text.write("The order of approximation is p = 10.\n" + (Style.RESET_ALL if use_colors else ""))
        return text.getvalue()

    def _fft_time_evolution(self):

        moment_space_wave_tensor = cp.fft.fftn(self.__wave_tensor, norm="ortho")

        self.__kinetic_operator_kernel(
            self.__kernel_grid_size,
            self.__kernel_block_size,
            (
                moment_space_wave_tensor,

                cp.float32(self.__delta_x_bohr_radii_3[0]),
                cp.float32(self.__delta_x_bohr_radii_3[1]),
                cp.float32(self.__delta_x_bohr_radii_3[2]),

                cp.float32(self.__delta_time_h_bar_per_hartree),
                cp.float32(self.__wave_packet.get_particle_mass_electron_rest_mass()),

                self.__wave_numbers[0],
                self.__wave_numbers[1],
                self.__wave_numbers[2]
            )
        )

        self.__wave_tensor = cp.fft.ifftn(moment_space_wave_tensor, norm="ortho")

        self.__potential_operator_kernel(
            self.__kernel_grid_size,
            self.__kernel_block_size,
            (
                self.__wave_tensor,
                self.__localised_potential_hartree,
                cp.float32(self.__delta_time_h_bar_per_hartree)
            )
        )

        moment_space_wave_tensor = cp.fft.fftn(self.__wave_tensor, norm="ortho")

        self.__kinetic_operator_kernel(
            self.__kernel_grid_size,
            self.__kernel_block_size,
            (
                moment_space_wave_tensor,

                cp.float32(self.__delta_x_bohr_radii_3[0]),
                cp.float32(self.__delta_x_bohr_radii_3[1]),
                cp.float32(self.__delta_x_bohr_radii_3[2]),

                cp.float32(self.__delta_time_h_bar_per_hartree),
                cp.float32(self.__wave_packet.get_particle_mass_electron_rest_mass()),

                self.__wave_numbers[0],
                self.__wave_numbers[1],
                self.__wave_numbers[2]
            )
        )

        self.__wave_tensor = cp.fft.ifftn(moment_space_wave_tensor, norm="ortho")

    """
    def _merged_fft_time_evolution(
            self,
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
    """

    def _power_series_time_evolution(self):
        shape = self.__wave_tensor.shape
        grid_size = (64, 64, 64)
        block_size = (shape[0] // grid_size[0], shape[1] // grid_size[1], shape[2] // grid_size[2])
        pingpong_idx = 0
        self.__pingpong_buffer[1 - pingpong_idx] = cp.copy(self.__wave_tensor)
        for n in range(1, self.__order_of_approximation + 1):
            self.__next_s_kernel(
                grid_size,
                block_size,
                (
                    self.__pingpong_buffer[1 - pingpong_idx],
                    self.__pingpong_buffer[pingpong_idx],
                    self.__localised_potential_hartree,
                    self.__wave_tensor,

                    cp.float32(self.__delta_time_h_bar_per_hartree),

                    cp.float32(self.__delta_x_bohr_radii_3[0]),
                    cp.float32(self.__delta_x_bohr_radii_3[1]),
                    cp.float32(self.__delta_x_bohr_radii_3[2]),

                    cp.float32(self.__wave_packet.get_particle_mass_electron_rest_mass()),
                    cp.int32(n)
                )
            )
            pingpong_idx = 1 - pingpong_idx

    def evolve_state(self):
        if self.__simulation_method == SimulationMethod.FOURIER:
            self._fft_time_evolution()
        elif self.__simulation_method == SimulationMethod.POWER_SERIES:
            self._power_series_time_evolution()
