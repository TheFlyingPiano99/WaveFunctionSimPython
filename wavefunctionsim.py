import numpy as np
import sources.math_utils as math_utils
from sources import wave_packet, volume_visualization, animation, measurement, potential
import toml

def init_kinetic_operator(N, delta_x, delta_time):
    try:
        return np.load(file='cache/kinetic_operator.npy')
    except OSError:
        print('No cached kinetic_operator.npy found.')
    P_kinetic = np.zeros(shape=(N, N, N), dtype=np.complex_)
    for x in range(0, N):
        for y in range(0, N):
            for z in range(0, N):
                f = np.array([x, y, z]) / np.array([N, N, N])
                # Fix numpy fftn-s "negative frequency in second half issue"
                if (f[0] > 0.5):
                    f[0] = 1.0 - f[0]
                if (f[1] > 0.5):
                    f[1] = 1.0 - f[1]
                if (f[2] > 0.5):
                    f[2] = 1.0 - f[2]
                k = 2.0 * np.pi * f / delta_x
                angle = np.dot(k, k) * delta_time / 4.0
                P_kinetic[x, y, z] = math_utils.exp_i(angle)
    np.save(file='cache/kinetic_operator.npy', arr=P_kinetic)
    return P_kinetic

def init_potential_operator(V, N, delta_time):
    P_potential = np.zeros(shape=(N, N, N), dtype=np.complex_)
    for x in range(0, N):
        for y in range(0, N):
            for z in range(0, N):
                angle = - V[x, y, z] * delta_time
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

    with open("config/parameters.toml") as f:
        config = toml.load(f)


    # Maximal kinetic energy

    particle_mass = config['particle_mass']
    initial_velocity = config['initial_velocity']

    print(f"Mass of the particle is {particle_mass} electron rest mass.\n"
          f"Initial velocity of the particle is {initial_velocity} Bohr radius hartree / h-bar")

    particle_momentum_h_per_bohr_radius = math_utils.classical_momentum(mass=particle_mass, velocity=initial_velocity)
    print(f"Initial momentum of particle is {particle_momentum_h_per_bohr_radius} h-bar / Bohr radius")
    de_broglie_wave_length_bohr_radii = math_utils.get_de_broglie_wave_length_bohr_radii(particle_momentum_h_per_bohr_radius)
    print(f"De Broglie wavelength associated with the particle is {de_broglie_wave_length_bohr_radii} Bohr radii.")

    simulated_volume_width_bohr_radii = config['simulated_volume_width_bohr_radii']
    print(f"Width of simulated volume is w = {simulated_volume_width_bohr_radii} Bohr radii.")

    N = config['number_of_samples_per_axis']
    print(f"Number of samples per axis is N = {N}.")

    # Space resolution
    delta_x_bohr_radii = simulated_volume_width_bohr_radii / N
    print(f"Space resolution is delta_x = {delta_x_bohr_radii} Bohr radii.")
    if (delta_x_bohr_radii >= de_broglie_wave_length_bohr_radii / 2.0):
        print("WARNING: delta_x exceeds half of de Broglie wavelength!")

    # The maximum allowed delta_time
    upper_limit_on_delta_time_h_per_hartree = 4.0 / np.pi * (3.0 * delta_x_bohr_radii * delta_x_bohr_radii) / 3.0 # Based on reasoning from the Web-Schrödinger paper
    print(f"The maximal viable time resolution < {upper_limit_on_delta_time_h_per_hartree} h-bar / hartree")

    # Time increment of simulation
    delta_time_h_bar_per_hartree = 0.1 * upper_limit_on_delta_time_h_per_hartree
    print(f"Time resolution is delta = {delta_time_h_bar_per_hartree} h-bar / hartree.")

    initial_position_bohr_radii_3 = np.array([N, N, N]) / 2.0 * delta_x_bohr_radii - np.array([0, 0, 7.0])

    print("***************************************************************************************")

    print("Initializing wave packet")
    packet_width_bohr_radii = config['packet_width_bohr_radii']
    print(f"Wave packet width is {packet_width_bohr_radii} bohr radii.")
    a = packet_width_bohr_radii * 2.0
    wave_tensor = wave_packet.init_gaussian_wave_packet(N=N, delta_x_bohr_radii=delta_x_bohr_radii, a=a, r_0_bohr_radii_3=initial_position_bohr_radii_3,
                                            initial_momentum_h_per_bohr_radius_3=np.array([0, 0, -particle_momentum_h_per_bohr_radius]))

    # Normalize:
    probability_density = np.square(np.abs(wave_tensor))
    sum_probability = np.sum(probability_density)
    print(f"Sum of probabilities = {sum_probability}")
    wave_tensor = wave_tensor / (sum_probability ** 0.5)
    probability_density = np.square(np.abs(wave_tensor))
    sum_probability = np.sum(probability_density)
    print(f"Sum of probabilities after normalization = {sum_probability}")
    # Operators:
    print("Initializing kinetic energy operator")
    kinetic_operator = init_kinetic_operator(N=N, delta_x=delta_x_bohr_radii, delta_time=delta_time_h_bar_per_hartree)
    print("Initializing potential energy operator")
    try:
        V = np.load(file='cache/localized_potential.npy')
    except OSError:
        print('No cached localized_potential.npy found.')
        V = potential.init_potential_box(N=N, delta_x=delta_x_bohr_radii, wall_thickness_bohr_radii=3.0, potential_wall_height_hartree=1000.0)
        V = potential.add_single_slit(V=V, delta_x=delta_x_bohr_radii, center_bohr_radii=15.0, thickness_bohr_radii=1.0, height_hartree=1000.0, slit_size_bohr_radii=2.0)
        np.save(file='cache/localized_potential.npy', arr=V)

    try:
        potential_operator = np.load(file='cache/potential_operator.npy')
    except OSError:
        print('No cached potential_operator.npy found.')
        potential_operator = init_potential_operator(V=V, N=N, delta_time=delta_time_h_bar_per_hartree)
        np.save(file='cache/potential_operator.npy', arr=potential_operator)

    print("***************************************************************************************")
    print("Starting simulation")
    canvas = volume_visualization.VolumeCanvas(volume_data=probability_density, secondary_data=V)
    animation_writer = animation.AnimationWriter("output/probability_density_time_development.gif")
    measurement_plane = measurement.MeasurementPlane(wave_tensor=wave_tensor, delta_x=delta_x_bohr_radii, location_bohr_radii=25.0)
    animation_frame_step_interval = 5
    png_step_interval = 10
    measurement_plane_capture_interval = 10

    # Run simulation
    for i in range(1000):
        print("Iteration: ", i, ".")
        wave_tensor = time_evolution(wave_tensor= wave_tensor, kinetic_operator=kinetic_operator, potential_operator=potential_operator)
        probability_density = np.square(np.abs(wave_tensor))
        print(f"Integral of probability density P = {np.sum(probability_density)}.")
        canvas.update(probability_density)
        measurement_plane.integrate(wave_tensor=wave_tensor, delta_time=delta_time_h_bar_per_hartree)
        if (i % animation_frame_step_interval == 0):
            animation_writer.add_frame(canvas)
        if (i % png_step_interval == 0):
            canvas.save_to_png(f"output/probability_density_{i:3d}.png")
        if (i % measurement_plane_capture_interval == 0):
            measurement_plane.save(probability_save_path=f'output/measurement_plane_probability_{i:3d}.png', dwell_time_save_path=f'output/measurement_plane_dwell_time_{i:3d}.png')

    animation_writer.finish()
    print("Simulation has finished.")


if __name__=="__main__":
    sim()