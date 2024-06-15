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
    __total_iteration_count = 1000
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
    __absorbing_boundary_condition: potential.AbsorbingBoundaryCondition = None
    __potential_walls: list[potential.PotentialWall] = []
    __is_dynamic_potential_mode: bool = False
    __simulation_method: SimulationMethod = SimulationMethod.FOURIER
    __pre_initialized_potential: potential.PreInitializedPotential
    def __init__(self, config):

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

        self.__total_iteration_count = try_read_param(config, "simulation.total_iteration_count", 1000)
        self.__is_dynamic_potential_mode = try_read_param(config, "simulation.is_dynamic_potential_mode", True)
        self.__enable_wave_function_saving = try_read_param(config, "simulation.enable_wave_function_saving", False)
        self.__double_precision_calculation = try_read_param(config, "simulation.double_precision_calculation", False)

        # Wace packet:
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
                    self.__observation_box_bottom_corner_bohr_radii_3[i]
                    > self.__observation_box_top_corner_bohr_radii_3[i]
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
        self.__absorbing_boundary_condition = potential.AbsorbingBoundaryCondition(config)  # It processes the viewing boundaries
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
                )
            )
            cp.save(file=os.path.join(self.__cache_dir, "gaussian_wave_packet.npy"), arr=self.__wave_tensor)
        # Normalize:
        self.__probability_density = cp.asnumpy(cp.square(cp.abs(self.__wave_tensor)))
        sum_probability = cp.sum(self.__probability_density)
        print(f"Sum of probabilities = {sum_probability:.8f}")
        self.__wave_tensor = self.__wave_tensor / (sum_probability ** 0.5)
        self.__probability_density = cp.asnumpy(cp.square(cp.abs(self.__wave_tensor)))
        sum_probability = cp.sum(self.__probability_density)
        print(f"Sum of probabilities after normalization = {sum_probability:.8f}")

        if self.__simulation_method == SimulationMethod.FOURIER:
            # Operators:
            print("")
            print(Fore.GREEN + "Initializing kinetic energy operator" + Style.RESET_ALL)

            full_init = True
            if self.__use_cache:
                try:
                    self.__kinetic_operator = cp.asarray(
                        np.load(file=os.path.join(self.__cache_dir, "kinetic_operator.npy")))
                    full_init = False
                except OSError:
                    print("No cached kinetic_operator.npy found.")
            if full_init:
                self.__kinetic_operator = operators.init_kinetic_operator(
                    delta_x_3=self.__delta_x_bohr_radii_3,
                    delta_time=self.__delta_time_h_bar_per_hartree,
                    shape=self.__number_of_voxels_3
                )
                cp.save(file=os.path.join(self.__cache_dir, "kinetic_operator.npy"), arr=self.__kinetic_operator)

            print("")
            print(Fore.GREEN + "Initializing potential energy operator" + Style.RESET_ALL)
            print("")
            print(self.get_potential_description_text(use_colors=True))
        elif self.__simulation_method == SimulationMethod.POWER_SERIES:
            pass

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
                shape=self.__number_of_voxels_3, dtype=cp.csingle
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

            """
            print("Creating draining potential.")
            dp = sim_state.drain_potential_description
            sim_state.localised_potential_hartree = potential.add_draining_potential(
                V=sim_state.localised_potential_hartree,
                delta_x_3=sim_state.delta_x_bohr_radii_3,
                inner_radius_bohr_radii=dp.inner_radius_bohr_radii,
                outer_radius_bohr_radii=dp.outer_radius_bohr_radii,
                max_potential_hartree=dp.max_potential_hartree,
                exponent=dp.exponent,
            )
            """

            """
            try:
                interaction = self.__config["particle_hard_interaction"]
                r = interaction["particle_radius_bohr_radii"]
                v = interaction["potential_hartree"]
                print("Creating particle hard interaction potential.")
                tensor = potential.particle_hard_interaction_potential(
                    delta_x_3=self.__delta_x_bohr_radii_3,
                    particle_radius_bohr_radius=r,
                    potential_hartree=v,
                    V=cp.zeros(shape=self.__number_of_voxels_3, dtype=cp.csingle),
                )
                self.__localised_potential_hartree += tensor
                try:
                    visible = interaction["visible"]
                    if visible:
                        self.__localised_potential_to_visualize_hartree += tensor
                except KeyError:
                    pass
            except KeyError:
                pass
            """

            """
            try:
                interaction = self.__config["particle_inv_squared_interaction"]
                v = interaction["center_potential_hartree"]
                print("Creating particle inverse square interaction potential.")
                tensor = potential.particle_inv_square_interaction_potential(
                    delta_x_3=self.__delta_x_bohr_radii_3,
                    potential_hartree=v,
                    V=cp.zeros(shape=self.__number_of_voxels_3, dtype=cp.csingle),
                )
                self.__localised_potential_hartree += tensor
                try:
                    visible = interaction["visible"]
                    if visible:
                        self.__localised_potential_to_visualize_hartree += tensor
                except KeyError:
                    pass
            except KeyError:
                pass
            """

            """
            try:
                oscillator = self.__config["harmonic_oscillator_1d"]
                omega = oscillator["angular_frequency_radian_hartree_per_h_bar"]
                print("Creating harmonic oscillator.")
                tensor = potential.add_harmonic_oscillator_for_1D(
                    delta_x_3=self.__delta_x_bohr_radii_3,
                    angular_frequency=omega,
                    V=cp.zeros(shape=self.__number_of_voxels_3, dtype=cp.csingle),
                )
                self.__localised_potential_hartree += tensor
                try:
                    visible = oscillator["visible"]
                    if visible:
                        self.__localised_potential_to_visualize_hartree += tensor
                except KeyError:
                    pass
            except KeyError:
                pass

            """

            """
            try:
                walls_arr = self.__config["walls_1d"]
                for wall_1d in walls_arr:
                    c = wall_1d["center_bohr_radii"]
                    v = wall_1d["potential_hartree"]
                    t = wall_1d["thickness_bohr_radii"]
                    print("Creating wall.")
                    tensor = potential.add_wall_for_1D(
                        delta_x_3=self.__delta_x_bohr_radii_3,
                        potential_hartree=v,
                        thickness_bohr_radius=t,
                        center_bohr_radius=c,
                        V=cp.zeros(shape=self.__number_of_voxels_3, dtype=cp.csingle),
                    )
                    self.__localised_potential_hartree += tensor
                    try:
                        visible = wall_1d["visible"]
                        if visible:
                            self.__localised_potential_to_visualize_hartree += tensor
                    except KeyError:
                        pass
            except KeyError:
                pass
            """

            for wall in self.__potential_walls:
                print("Creating wall.")
                tensor = wall.add_potential(
                    delta_x_3=self.__delta_x_bohr_radii_3,
                    V=cp.zeros(shape=self.__number_of_voxels_3, dtype=cp.csingle),
                )
                self.__localised_potential_hartree += tensor
                if wall.visible:
                    self.__localised_potential_to_visualize_hartree += tensor

            """
            try:
                grid_arr = self.__config["optical_grids"]
                for grid in grid_arr:
                    v = grid["potential_hartree"]
                    c = np.array(grid["center_bohr_radii_3"], dtype=float)
                    n = math_utils.normalize(np.array(grid["normal_vector_3"], dtype=float))
                    d = grid["distance_between_nodes_bohr_radii"]
                    i = grid["node_in_one_direction"]
                    print("Creating optical grid.")
                    tensor = potential.add_optical_grid(
                        delta_x_3=self.__delta_x_bohr_radii_3,
                        potential_hartree=v,
                        distance_between_nodes_bohr_radii=d,
                        center_bohr_radius=c,
                        normal=n,
                        node_count=i,
                        V=cp.zeros(shape=self.__number_of_voxels_3, dtype=cp.csingle),
                    )
                    self.__localised_potential_hartree += tensor
                    try:
                        visible = grid["visible"]
                        if visible:
                            self.__localised_potential_to_visualize_hartree += tensor
                    except KeyError:
                        pass
            except KeyError:
                pass
            """

            """
            try:
                ds_array = self.__config["double_slits"]
                for double_slit in ds_array:
                    print("Creating double-slit.")
                    space_between_slits = double_slit["distance_between_slits_bohr_radii"]
                    tensor = potential.add_double_slit(
                        V=cp.zeros(shape=self.__number_of_voxels_3, dtype=cp.csingle),
                        delta_x_3=self.__delta_x_bohr_radii_3,
                        center_bohr_radii_3=np.array(double_slit["center_bohr_radius_3"]),
                        thickness_bohr_radii=double_slit["thickness_bohr_radii"],
                        potential_hartree=double_slit["potential_hartree"],
                        slit_width_bohr_radii=double_slit["slit_width_bohr_radii"],
                        space_between_slits_bohr_radii=space_between_slits,
                    )
                    self.__localised_potential_hartree += tensor
                    try:
                        visible = double_slit["visible"]
                        if visible:
                            self.__localised_potential_to_visualize_hartree += tensor
                    except KeyError:
                        pass
            except KeyError:
                pass
            """

            """
            try:
                coulomb_potential = self.__config["coulomb_potential"]
                print("Creating Coulomb potential.")
                tensor = potential.add_coulomb_potential(
                    V=cp.zeros(shape=self.__number_of_voxels_3, dtype=cp.csingle),
                    delta_x_3=self.__delta_x_bohr_radii_3,
                    center_bohr_radius=np.array(coulomb_potential["center_bohr_radii_3"]),
                    gradient_dir=np.array(coulomb_potential["gradient_direction"]),
                    charge_density=coulomb_potential["charge_density_elementary_charge_per_bohr_radius"],
                    oxide_start_bohr_radii=coulomb_potential["oxide_start_bohr_radii"],
                    oxide_end_bohr_radii=coulomb_potential["oxide_end_bohr_radii"]
                )
                self.__localised_potential_hartree += tensor
                visible = coulomb_potential["visible"]
                if visible:
                    self.__localised_potential_to_visualize_hartree += tensor
            except KeyError:
                print("No Coulomb potential created")
            """

            """
            try:
                gradient = self.__config["linear_potential_gradient"]
                print("Creating linear potential gradient.")
                tensor = potential.add_linear_potential_gradient(
                    V=cp.zeros(shape=self.__number_of_voxels_3, dtype=cp.csingle),
                    delta_x_3=self.__delta_x_bohr_radii_3,
                    center_bohr_radius=np.array(gradient["center_bohr_radii_3"]),
                    gradient_dir=np.array(gradient["gradient_direction"]),
                    gradient_val=gradient["gradient_magnitude_hartree_per_bohr_radius"],
                )
                self.__localised_potential_hartree += tensor
                visible = gradient["visible"]
                if visible:
                    self.__coulomb_potential = tensor
            except KeyError:
                pass
            """

            np.save(
                file=os.path.join(self.__cache_dir, "localized_potential.npy"),
                arr=cp.asnumpy(self.__localised_potential_hartree),
            )
            np.save(file=os.path.join(self.__cache_dir, "localized_potential_to_visualize.npy"),
                    arr=cp.asnumpy(self.__localised_potential_to_visualize_hartree))

        full_init = True
        if self.__simulation_method == SimulationMethod.FOURIER:
            if self.__use_cache:
                try:
                    self.__potential_operator = cp.load(
                        file=os.path.join(self.__cache_dir, "potential_operator.npy"))
                    full_init = False
                except OSError:
                    print("No cached potential_operator.npy found.")
            if full_init:
                print("Creating potential operator.")
                self.__potential_operator = cp.zeros(shape=self.__localised_potential_hartree.shape,
                                                        dtype=self.__localised_potential_hartree.dtype)
                self.__potential_operator = operators.init_potential_operator(
                    P_potential=self.__potential_operator,
                    V=self.__localised_potential_hartree,
                    delta_time=self.__delta_time_h_bar_per_hartree,
                )

                cp.save(file=os.path.join(self.__cache_dir, "potential_operator.npy"),
                        arr=self.__potential_operator)


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

    def get_view_into_raw_wave_function(self):
        return math_utils.cut_window(
            arr=self.__wave_tensor,
            bottom=self.__observation_box_bottom_corner_voxel_3,
            top=self.__observation_box_top_corner_voxel_3,
        )


    def get_view_into_probability_density(self):
        return math_utils.cut_window(
            arr=self.__probability_density,
            bottom=self.__observation_box_bottom_corner_voxel_3,
            top=self.__observation_box_top_corner_voxel_3,
        )

    def get_view_into_potential(self):
        return math_utils.cut_window(
            arr=cp.real(self.__localised_potential_to_visualize_hartree),
            bottom=self.__observation_box_bottom_corner_voxel_3,
            top=self.__observation_box_top_corner_voxel_3,
        )

    def get_view_into_complex_potential(self):
        return math_utils.cut_window(
            arr=self.__localised_potential_to_visualize_hartree,
            bottom=self.__observation_box_bottom_corner_voxel_3,
            top=self.__observation_box_top_corner_voxel_3,
        )

    def get_view_into_coulomb_potential(self):
        return math_utils.cut_window(
            arr=self.__coulomb_potential,
            bottom=self.__observation_box_bottom_corner_voxel_3,
            top=self.__observation_box_top_corner_voxel_3,
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

        if self.__simulation_method == "fft":
            self.potential_operator = sources.operators.init_potential_operator(
                P_potential=self.__potential_operator,
                V=self.__localised_potential_hartree,
                delta_time=self.__delta_time_h_bar_per_hartree,
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

        """
        try:
            slits = self.__config["double_slits"]
            text.write(
                "Double slits:\n"
            )
            for slit in slits:
                wall_potential = slit["potential_hartree"]
                self.write_potential_wall_warnings(text, wall_potential, slit, use_colors)

                space_between_slits = slit["distance_between_slits_bohr_radii"]
                if space_between_slits > self.__de_broglie_wave_length_bohr_radii:
                    text.write(
                        (Fore.RED if use_colors else "")
                        + f"WARNING: Space between slits = {space_between_slits} exceeds the de Brogile wavelength = {self.__de_broglie_wave_length_bohr_radii:.4f} of the particle!"
                        + (Style.RESET_ALL if use_colors else "")
                    )
                text.write("\n")
        except KeyError:
            pass

        try:
            walls = self.__config["walls"]
            text.write(
                "Walls:\n"
            )
            for i, wall in enumerate(walls):
                text.write(
                    f"{i + 1}.\n"
                )
                wall_potential = wall["potential_hartree"]
                self.write_potential_wall_warnings(text, wall_potential, wall, use_colors)
                thickness = wall["thickness_bohr_radii"]
                text.write(f"Wall thickness is {thickness:.2f} Bohr radii.\n")
                text.write("\n")
        except KeyError:
            pass

        try:
            walls = self.__config["walls_1d"]
            text.write(
                "1D walls:\n"
            )
            for i, wall in enumerate(walls):
                text.write(
                    f"{i + 1}.\n"
                )
                wall_potential = wall["potential_hartree"]
                self.write_potential_wall_warnings(text, wall_potential, wall, use_colors)
                thickness = wall["thickness_bohr_radii"]
                text.write(f"Wall thickness is {thickness:.2f} Bohr radii.\n")
                center = wall["center_bohr_radii"]
                text.write(f"Wall center is at the x = {center:.2f} Bohr radius coordinate.\n")
                text.write("\n")
        except KeyError:
            pass

        try:
            interaction = self.__config["particle_hard_interaction"]
            text.write("1D particle hard interaction potential:\n")
            v = interaction["potential_hartree"]
            text.write(f"Potential is {v} Hartree.\n")
            r = interaction["particle_radius_bohr_radii"]
            text.write(f"Particle radius is {r} Bohr radii.\n")
            text.write("\n")
        except KeyError:
            pass

        try:
            interaction = self.__config["particle_inv_squared_interaction"]
            text.write("1D particle inverse squared interaction potential:\n")
            v = interaction["central_potential_hartree"]
            text.write(f"Potential at unit distance is {v} Hartree.\n")
            text.write("\n")
        except KeyError:
            pass


        try:
            oscillator = self.__config["harmonic_oscillator_1d"]
            text.write("1D particle harmonic oscillator:\n")
            omega = oscillator["angular_frequency"]
            text.write(f"Angular velocity is {omega} radian * Hartree / h-bar.\n")
            text.write("\n")
        except KeyError:
            pass

        drain_max = self.__config["drain"]["outer_potential_hartree"]
        text.write(
            f"Draining potential value at the outer edge is {drain_max:.1f} Hartree.\n"
        )
        exponent = self.__config["drain"]["interpolation_exponent"]
        text.write(f"Draining potential exponent is {exponent:.1f}.\n")
        """
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
