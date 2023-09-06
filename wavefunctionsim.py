import numpy as np
import sources.math_utils as math_utils
import sources.wave_packet as wp
import sources.potential as potential
import sources.plot as plot

def init_kinetic_operator(N, delta_x, delta_time):
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
                k = f / delta_x

                angle = np.dot(k, k) * delta_time / 4.0
                P_kinetic[x, y, z] = math_utils.exp_i(angle)
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

    # Maximal kinetic energy

    particle_mass = math_utils.electron_rest_mass
    initial_velocity = 10.0

    print(f"Mass of the particle is {particle_mass} electron rest mass.\n"
          f"Initial velocity of the particle is {initial_velocity} Bohr radius hartree / h-bar")

    particle_momentum_h_per_bohr_radius = math_utils.classical_momentum(mass=particle_mass, velocity=initial_velocity)
    print(f"Initial momentum of particle is {particle_momentum_h_per_bohr_radius} h-bar / Bohr radius")
    de_broglie_wave_length_bohr_radii = math_utils.get_de_broglie_wave_length_bohr_radii(particle_momentum_h_per_bohr_radius)
    print(f"De Broglie wavelength associated with the particle is {de_broglie_wave_length_bohr_radii} Bohr radii.")

    simulated_volume_width_bohr_radii = 16.0
    print(f"Width of simulated volume is w = {simulated_volume_width_bohr_radii} Bohr radii.")

    N = 32
    print(f"Number of samples per axis is N = {N}.")

    # Space resolution
    delta_x_bohr_radii = simulated_volume_width_bohr_radii / N
    print(f"Space resolution is delta_x = {delta_x_bohr_radii} Bohr radii.")
    if (delta_x_bohr_radii >= de_broglie_wave_length_bohr_radii):
        print("WARNING: delta_x exceeds de Broglie wavelength!")

    # The maximum allowed delta_time
    max_delta_time_h_per_hartree = 4.0 / np.pi * (3.0 * delta_x_bohr_radii * delta_x_bohr_radii) / 3.0 # Based on reasoning from the Web-Schrödinger paper
    print(f"The maximal viable time resolution < {max_delta_time_h_per_hartree} h-bar / hartree")

    # Time increment of simulation
    delta_time_h_per_hartree = 0.75 * max_delta_time_h_per_hartree
    print(f"Time resolution is delta = {delta_time_h_per_hartree} h-bar / hartree.")

    print("***************************************************************************************")

    print("Initializing wave packet")
    packet_width_bohr_radii = 1.0
    print(f"Wave packet width is {packet_width_bohr_radii} bohr radii.")
    a = packet_width_bohr_radii * 2.0
    wave_tensor = wp.init_gaussian_wave_packet(N=N, delta_x_bohr_radii=delta_x_bohr_radii, a=a, r_0_bohr_radii_3=np.array([N, N, N]) / 2.0 * delta_x_bohr_radii,
                                            initial_momentum_h_per_bohr_radius_3=np.array([0, 0, particle_momentum_h_per_bohr_radius]))

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
    kinetic_operator = init_kinetic_operator(N=N, delta_x=delta_x_bohr_radii, delta_time=delta_time_h_per_hartree)
    print("Initializing potential energy operator")
    V = potential.init_potential_box(N=N, delta_x=delta_x_bohr_radii, wall_thickness_bohr_radii=2.0, potential_wall_height_hartree=10.0)
    potential_operator = init_potential_operator(V=V, N=N, delta_time=delta_time_h_per_hartree)
    plot.plot_potential_image(V=V, N=N, delta_x=delta_x_bohr_radii)

    print("***************************************************************************************")
    print("Starting simulation")

    # Run simulation
    for i in range(1000):
        print("Iteration: ", i, ".")
        wave_tensor = time_evolution(wave_tensor= wave_tensor, kinetic_operator=kinetic_operator, potential_operator=potential_operator)
        probability_density = np.square(np.abs(wave_tensor))
        print(f"Integral of probability density P = {np.sum(probability_density)}.")
        plot.plot_probability_density_image(probability_density=probability_density, delta_time_h_per_hartree=delta_time_h_per_hartree, delta_x=delta_x_bohr_radii, N=N, i=i)

    print("Simulation has finished.")


sim()