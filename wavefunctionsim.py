import numpy as np
import sources.math_utils as math_utils
from sources import wave_packet, volume_visualization, animation, measurement, potential
import toml
import time
import sources.plot as plot
from numba import jit


@jit(nopython=True)
def init_kinetic_operator(N, delta_x, delta_time):
    P_kinetic = np.zeros(shape=(N, N, N), dtype=np.complex_)
    for x in range(0, N):
        for y in range(0, N):
            for z in range(0, N):
                f = np.array([x, y, z]) / np.array([N, N, N])
                # Fix numpy fftn-s "negative frequency in second half issue"
                if f[0] > 0.5:
                    f[0] = 1.0 - f[0]
                if f[1] > 0.5:
                    f[1] = 1.0 - f[1]
                if f[2] > 0.5:
                    f[2] = 1.0 - f[2]
                k = 2.0 * np.pi * f / delta_x
                angle = np.dot(k, k) * delta_time / 4.0
                P_kinetic[x, y, z] = math_utils.exp_i(angle)
    return P_kinetic


@jit(nopython=True)
def init_potential_operator(V, N, delta_time):
    P_potential = np.zeros(shape=(N, N, N), dtype=np.complex_)
    for x in range(0, N):
        for y in range(0, N):
            for z in range(0, N):
                angle = -V[x, y, z] * delta_time
                P_potential[x, y, z] = math_utils.exp_i(angle)
    return P_potential


def time_evolution(wave_tensor, kinetic_operator, potential_operator):
    moment_space_wave_tensor = np.fft.fftn(wave_tensor, norm="forward")
    moment_space_wave_tensor = np.multiply(kinetic_operator, moment_space_wave_tensor)
    wave_tensor = np.fft.fftn(moment_space_wave_tensor, norm="backward")
    wave_tensor = np.multiply(potential_operator, wave_tensor)
    moment_space_wave_tensor = np.fft.fftn(wave_tensor, norm="forward")
    moment_space_wave_tensor = np.multiply(kinetic_operator, moment_space_wave_tensor)
    return np.fft.fftn(moment_space_wave_tensor, norm="backward")


def sim():
    print("Wave function simulation")
    # We use hartree atomic unit system
    initialisation_start_time_s = time.time()
    with open("config/parameters.toml") as f:
        config = toml.load(f)

    # Maximal kinetic energy

    particle_mass = config["particle_mass"]
    initial_wp_velocity_bohr_radii_hartree_per_h_bar = np.array(
        config["initial_wp_velocity_bohr_radii_hartree_per_h_bar"]
    )
    velocity_magnitude = (
        np.dot(
            initial_wp_velocity_bohr_radii_hartree_per_h_bar,
            initial_wp_velocity_bohr_radii_hartree_per_h_bar,
        )
        ** 0.5
    )
    print(
        f"Mass of the particle is {particle_mass} electron rest mass.\n"
        f"Initial velocity of the particle is {velocity_magnitude} Bohr radius hartree / h-bar"
    )

    initial_wp_momentum_h_per_bohr_radius = math_utils.classical_momentum(
        mass=particle_mass, velocity=initial_wp_velocity_bohr_radii_hartree_per_h_bar
    )
    momentum_magnitude = (
        np.dot(
            initial_wp_momentum_h_per_bohr_radius, initial_wp_momentum_h_per_bohr_radius
        )
        ** 0.5
    )
    print(
        f"Initial mean momentum of particle is {momentum_magnitude} h-bar / Bohr radius"
    )
    de_broglie_wave_length_bohr_radii = (
        math_utils.get_de_broglie_wave_length_bohr_radii(momentum_magnitude)
    )
    print(
        f"De Broglie wavelength associated with the particle is {de_broglie_wave_length_bohr_radii} Bohr radii."
    )

    initial_kinetic_energy_hartree = momentum_magnitude**2 / 2 / particle_mass
    print(
        f"Initial mean kinetic energy of the particle is {initial_kinetic_energy_hartree} hartree."
    )

    simulated_volume_width_bohr_radii = config["simulated_volume_width_bohr_radii"]
    print(
        f"Width of simulated volume is w = {simulated_volume_width_bohr_radii} Bohr radii."
    )

    N = config["number_of_samples_per_axis"]
    print(f"Number of samples per axis is N = {N}.")

    # Space resolution
    delta_x_bohr_radii = simulated_volume_width_bohr_radii / N
    print(f"Space resolution is delta_x = {delta_x_bohr_radii} Bohr radii.")
    if delta_x_bohr_radii >= de_broglie_wave_length_bohr_radii / 2.0:
        print("WARNING: delta_x exceeds half of de Broglie wavelength!")

    # The maximum allowed delta_time
    upper_limit_on_delta_time_h_per_hartree = (
        4.0 / np.pi * (3.0 * delta_x_bohr_radii * delta_x_bohr_radii) / 3.0
    )  # Based on reasoning from the Web-Schrödinger paper
    print(
        f"The maximal viable time resolution < {upper_limit_on_delta_time_h_per_hartree} h-bar / hartree"
    )

    # Time increment of simulation
    delta_time_h_bar_per_hartree = 0.1 * upper_limit_on_delta_time_h_per_hartree
    print(f"Time resolution is delta = {delta_time_h_bar_per_hartree} h-bar / hartree.")

    initial_wp_position_bohr_radii_3 = np.array(
        config["initial_wp_position_bohr_radii_3"]
    )

    print(
        "***************************************************************************************"
    )

    print("Initializing wave packet")
    wp_width_bohr_radii = config["wp_width_bohr_radii"]
    print(f"Wave packet width is {wp_width_bohr_radii} bohr radii.")
    a = wp_width_bohr_radii * 2.0
    try:
        wave_tensor = np.load(file="cache/gaussian_wave_packet.npy")
    except OSError:
        print("No cached gaussian_wave_packet.npy found.")
        wave_tensor = wave_packet.init_gaussian_wave_packet(
            N=N,
            delta_x_bohr_radii=delta_x_bohr_radii,
            a=a,
            r_0_bohr_radii_3=initial_wp_position_bohr_radii_3,
            initial_momentum_h_per_bohr_radius_3=-initial_wp_momentum_h_per_bohr_radius,
        )
        np.save(file="cache/gaussian_wave_packet.npy", arr=wave_tensor)

    # Normalize:
    probability_density = np.square(np.abs(wave_tensor))
    sum_probability = np.sum(probability_density)
    print(f"Sum of probabilities = {sum_probability}")
    wave_tensor = wave_tensor / (sum_probability**0.5)
    probability_density = np.square(np.abs(wave_tensor))
    sum_probability = np.sum(probability_density)
    print(f"Sum of probabilities after normalization = {sum_probability}")
    # Operators:
    print("Initializing kinetic energy operator")
    try:
        kinetic_operator = np.load(file="cache/kinetic_operator.npy")
    except OSError:
        print("No cached kinetic_operator.npy found.")
        kinetic_operator = init_kinetic_operator(
            N=N, delta_x=delta_x_bohr_radii, delta_time=delta_time_h_bar_per_hartree
        )
        np.save(file="cache/kinetic_operator.npy", arr=kinetic_operator)

    print("Initializing potential energy operator")
    try:
        V = np.load(file="cache/localized_potential.npy")
        only_the_obstacle_potential = np.load(
            file="cache/only_the_obstacle_potential.npy"
        )
    except OSError:
        print("No cached localized_potential.npy found.")
        V = np.zeros(shape=(N, N, N), dtype=float)
        # V = potential.add_single_slit(V=V, delta_x=delta_x_bohr_radii, center_bohr_radii=15.0, thickness_bohr_radii=1.0, height_hartree=200.0, slit_size_bohr_radii=3.0)
        V = potential.add_double_slit(
            V=V,
            delta_x=delta_x_bohr_radii,
            center_bohr_radii=np.array([0.0, 15.0, 15.0]),
            thickness_bohr_radii=1.5,
            height_hartree=200.0,
            slit_width_bohr_radii=2,
            space_between_slits_bohr_radii=0.5,
        )
        # V = potential.add_wall(V=V, delta_x=delta_x_bohr_radii, center_bohr_radii=15.0, thickness_bohr_radii=1.5, height_hartree=200)
        only_the_obstacle_potential = V.copy()
        V = potential.init_potential_box(
            V=V,
            N=N,
            delta_x=delta_x_bohr_radii,
            wall_thickness_bohr_radii=1.5,
            potential_wall_height_hartree=1000.0,
        )
        np.save(file="cache/localized_potential.npy", arr=V)
        np.save(
            file="cache/only_the_obstacle_potential.npy",
            arr=only_the_obstacle_potential,
        )

    try:
        potential_operator = np.load(file="cache/potential_operator.npy")
    except OSError:
        print("No cached potential_operator.npy found.")
        potential_operator = init_potential_operator(
            V=V, N=N, delta_time=delta_time_h_bar_per_hartree
        )
        np.save(file="cache/potential_operator.npy", arr=potential_operator)
    print(
        f"Time spent with initialisation: {time.time() - initialisation_start_time_s} s."
    )
    print(
        "***************************************************************************************"
    )
    print("Starting simulation")
    canvas = volume_visualization.VolumeCanvas(
        volume_data=probability_density, secondary_data=only_the_obstacle_potential
    )
    animation_writer = animation.AnimationWriter(
        "output/probability_density_time_development.gif"
    )

    measurement_plane = measurement.MeasurementPlane(
        wave_tensor=wave_tensor, delta_x=delta_x_bohr_radii, location_bohr_radii=25.0
    )
    measurement_volume_full = measurement.AAMeasurementVolume(
        bottom_corner=(0, 0, 0), top_corner=(N, N, N), label="Full volume"
    )
    measurement_volume_first_half = measurement.AAMeasurementVolume(
        bottom_corner=(0, 0, 0), top_corner=(N, N, int(N / 2)), label="First half"
    )
    measurement_volume_second_half = measurement.AAMeasurementVolume(
        bottom_corner=(0, 0, int(N / 2)), top_corner=(N, N, N), label="Second half"
    )

    animation_frame_step_interval = config["animation_frame_step_interval"]
    png_step_interval = config["png_step_interval"]
    measurement_plane_capture_interval = config["measurement_plane_capture_interval"]
    probability_plot_interval = config["probability_plot_interval"]
    elapsed_iter_time = 0.0
    total_iteration_count = config["total_iteration_count"]

    # Run simulation
    for i in range(total_iteration_count + 1):
        iter_start_time_s = time.time()
        print("Iteration: ", i, ".")
        probability_density = np.square(np.abs(wave_tensor))

        measurement_volume_full.integrate_probability_density(probability_density)
        measurement_volume_first_half.integrate_probability_density(probability_density)
        measurement_volume_second_half.integrate_probability_density(
            probability_density
        )

        print(
            f"Integral of probability density P = {measurement_volume_full.get_probability()}."
        )
        print(
            f"Probability of particle in first half P(first) = {measurement_volume_first_half.get_probability():.3}."
        )
        print(
            f"Probability of particle in second half P(second) = {measurement_volume_second_half.get_probability():.3}."
        )

        canvas.update(
            probability_density,
            iter_count=i,
            delta_time_h_bar_per_hartree=delta_time_h_bar_per_hartree,
        )
        measurement_plane.integrate(
            wave_tensor=wave_tensor, delta_time=delta_time_h_bar_per_hartree
        )
        if i % animation_frame_step_interval == 0:
            animation_writer.add_frame(canvas)
        if i % png_step_interval == 0:
            canvas.save_to_png(f"output/probability_density_{i:04d}.png")
        if i % probability_plot_interval == 0:
            plot.plot_probability_evolution(
                [
                    measurement_volume_full.get_probability_evolution(),
                    measurement_volume_first_half.get_probability_evolution(),
                    measurement_volume_second_half.get_probability_evolution(),
                ],
                delta_t=delta_time_h_bar_per_hartree,
                index=i,
                show_fig=i == 1000,
            )
        if i % measurement_plane_capture_interval == 0:
            measurement_plane.save(
                probability_save_path=f"output/measurement_plane_probability_{i:04d}.png",
                dwell_time_save_path=f"output/measurement_plane_dwell_time_{i:04d}.png",
            )

        wave_tensor = time_evolution(
            wave_tensor=wave_tensor,
            kinetic_operator=kinetic_operator,
            potential_operator=potential_operator,
        )

        iter_time = time.time() - iter_start_time_s
        elapsed_iter_time += iter_time
        print(f"Iteration time: {iter_time} s")

    animation_writer.finish()
    print("Simulation has finished.")
    print(f"Total simulation time: {elapsed_iter_time} s")
    print(
        f"Average iteration time: {elapsed_iter_time / float(total_iteration_count)} s"
    )


if __name__ == "__main__":
    sim()
