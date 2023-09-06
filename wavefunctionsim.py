import numpy as np
import plotly.graph_objects as go
import math


def exp_i(angle):
    return math.cos(angle) + 1j * math.sin(angle)

# This part is highly ambiguous
def P_free_space(r, t):
    return  1.0 / (2.0 * math.pi * t) ** 0.5 * math.exp(- 1j * math.pi / 4) * math.exp(1j * np.dot(r, r) / 2.0 / t)

def wave_0_x(x):
    sum = np.complex_(0.0)
    for i in range(10):
        sum += P_free_space(np.array([x, 0]), i)
    return sum

def wave_0_y(y):
    sum = np.complex_(0.0)
    for i in range(10):
        sum += P_free_space(np.array([0, y]), i)
    return sum

def wave_packet(x, y):
    return wave_0_x(x) * wave_0_y(y)

# End of the ambiguous part :)

def init_potential_box(N, delta_x, wall_thickness_bohr_radii, potential_wall_height_hartree):
    V = np.zeros(shape=(N, N, N), dtype=float)
    for x in range(0, N):
        for y in range(0, N):
            for z in range(0, N):
                r = np.array([x, y, z]) * delta_x
                V[x, y, z] = 0.0  # Zero in the middle
                # Barriers:
                # X-axis:
                if r[0] < wall_thickness_bohr_radii:
                    V[x, y, z] = potential_wall_height_hartree * (wall_thickness_bohr_radii - r[0]) / wall_thickness_bohr_radii
                if N * delta_x - r[0] < wall_thickness_bohr_radii:
                    V[x, y, z] = potential_wall_height_hartree * (wall_thickness_bohr_radii - (delta_x * N - r[0])) / wall_thickness_bohr_radii
                # Y-axis:
                if r[1] < wall_thickness_bohr_radii:
                    V[x, y, z] = potential_wall_height_hartree * (wall_thickness_bohr_radii - r[1]) / wall_thickness_bohr_radii
                if N * delta_x - r[1] < wall_thickness_bohr_radii:
                    V[x, y, z] = potential_wall_height_hartree * (wall_thickness_bohr_radii - (delta_x * N - r[1])) / wall_thickness_bohr_radii
                # Z-axis:
                if r[2] < wall_thickness_bohr_radii:
                    V[x, y, z] = potential_wall_height_hartree * (wall_thickness_bohr_radii - (r[2])) / wall_thickness_bohr_radii
                if N * delta_x - r[2] < wall_thickness_bohr_radii:
                    V[x, y, z] = potential_wall_height_hartree * (wall_thickness_bohr_radii - (delta_x * N - r[2])) / wall_thickness_bohr_radii
    return V


def init_potential_sphere(N, delta_x, wall_thickness, potential_wall_hight):
    V = np.zeros(shape=(N, N, N), dtype=float)
    for x in range(0, N):
        for y in range(0, N):
            for z in range(0, N):
                r = np.array([x, y, z]) / np.array([N - 1, N - 1, N - 1])
                V[x, y, z] = 0.0  # Zero in the middle
                # Barriers:
                dir = r - np.array([0.5, 0.5, 0.5])
                l = np.dot(dir, dir) ** 0.5
                if l > (0.5 - wall_thickness):
                    V[x, y, z] = potential_wall_hight * (l - (0.5 - wall_thickness)) / wall_thickness
    return V

def init_zero_potential(N):
    V = np.zeros(shape=(N, N, N), dtype=float)
    return V

def init_gaussian_wave_packet(N, delta_x_bohr_radii, a, r_0_bohr_radii_3, k_0_hartree):
    wave_tensor = np.zeros(shape=(N, N, N), dtype=np.complex_)
    for x in range(0, N):
        for y in range(0, N):
            for z in range(0, N):
                r = np.array([x, y, z]) * delta_x_bohr_radii
                wave_tensor[x, y, z] = (2.0 / math.pi / a**2) ** (3.0/4.0) * exp_i(np.dot(k_0_hartree, r)) * math.exp(- np.dot(r - r_0_bohr_radii_3, r - r_0_bohr_radii_3) / a ** 2)
    return wave_tensor

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
                P_kinetic[x, y, z] = exp_i(angle)
    return P_kinetic

def init_potential_operator(V, N, delta_time):
    P_potential = np.zeros(shape=(N, N, N), dtype=np.complex_)
    for x in range(0, N):
        for y in range(0, N):
            for z in range(0, N):
                angle = - V[x, y, z] * delta_time
                P_potential[x, y, z] = exp_i(angle)
    return P_potential


def time_evolution(wave_tensor, kinetic_operator, potential_operator):
    moment_space_wave_tensor = np.fft.fftn(wave_tensor, norm="forward")
    moment_space_wave_tensor = np.multiply(kinetic_operator, moment_space_wave_tensor)
    wave_tensor = np.fft.fftn(moment_space_wave_tensor, norm="backward")
    wave_tensor = np.multiply(potential_operator, wave_tensor)
    moment_space_wave_tensor = np.fft.fftn(wave_tensor, norm="forward")
    moment_space_wave_tensor = np.multiply(kinetic_operator, moment_space_wave_tensor)
    return np.fft.fftn(moment_space_wave_tensor, norm="backward")


def save_potential_image(V, volume_width):
    x, y, z = np.mgrid[:volume_width, :volume_width, :volume_width]
    fig = go.Figure()
    fig.add_trace(go.Volume(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=V.flatten(),
        isomin=0.0,
        isomax=10.0,
        opacity=0.3,
        surface_count=20,
        colorscale='Viridis',
        colorbar_title='V [hartree]'
    ))

    # Set the layout of the figure
    fig.update_layout(
        title='Potential energy',
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z')
        ),
        scene_camera_eye=dict(
            x=2.0 * math.sin(0),
            y=2.0 * math.cos(0),
            z=1.0
        )
    )
    # Show the figure
    fig.write_image(f"images/potential_energy.jpeg")


def save_probability_density_image(probability_density, volume_width, i : int):
    x, y, z = np.mgrid[:volume_width, :volume_width, :volume_width]
    fig = go.Figure()
    fig.add_trace(go.Volume(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value = probability_density.flatten(),
        isomin=0.0,
        isomax= 0.01 / (volume_width * volume_width * volume_width),
        opacity=0.3,
        surface_count=50,
        colorscale='Viridis',
        colorbar_title='Probability density'
    ))

    # Set the layout of the figure
    fig.update_layout(
        title='Probability density (time step = {})'.format(i),
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z')
        ),
        scene_camera_eye=dict(
            x=2.0 * math.sin(i / 200.0 * math.pi),
            y=2.0 * math.cos(i / 200.0 * math.pi),
            z=1.0
        )
    )
    # Show the figure
    fig.write_image(f"images/probability_density{i:03d}.jpeg")

def electron_volt_to_hartree(value):
    return value * 3.67493 * 10.0**(-2)

def angstrom_to_bohr_radii(value):
    return value * 0.52917721067

def sim():
    print("Wave function simulation")
    # We use hartree atomic unit system

    # Maximal kinetic energy

    E_max_hartree = electron_volt_to_hartree(5)
    print(f"Maximal kinetic energy is {E_max_hartree} hartree")

    de_broglie_wave_length_bohr_radii = angstrom_to_bohr_radii(5.5)
    print(f"Current  de Broglie wavelength is {de_broglie_wave_length_bohr_radii} Bohr radii.")

    simulated_volume_width_bohr_radii = 30.0
    print(f"Width of simulated volume is w = {simulated_volume_width_bohr_radii} Bohr radii.")

    N = 32
    print(f"Number of discrete space chunks per axis is N = {N}.")

    # Space resolution
    delta_x_bohr_radii = simulated_volume_width_bohr_radii / N
    print(f"Space resolution is delta_x = {delta_x_bohr_radii} Bohr radii.")

    # The maximum allowed delta_time
    max_delta_time_h_per_hartree = 4.0 / np.pi * 3.0 * delta_x_bohr_radii * delta_x_bohr_radii / 3.0 # Based on reasoning from the Web-Schrödinger paper
    print(f"The maximal viable time resolution < {max_delta_time_h_per_hartree} h-bar / hartree")

    # Time increment of simulation
    delta_time_h_per_hartree = 0.1 * max_delta_time_h_per_hartree
    print(f"Time resolution is delta_time = {delta_time_h_per_hartree} h-bar / hartree.")

    print("***************************************************************************************")

    print("Initializing wave-tensor")
    packet_width_bohr_radii = 4.0
    a = packet_width_bohr_radii * 2.0
    wave_tensor = init_gaussian_wave_packet(N=N, delta_x_bohr_radii=delta_x_bohr_radii, a=a, r_0_bohr_radii_3=np.array([N, N, N]) / 2.0 * delta_x_bohr_radii,
                                            k_0_hartree=np.array([0, 0, electron_volt_to_hartree(5)]))

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
    V = init_potential_box(N=N, delta_x=delta_x_bohr_radii, wall_thickness_bohr_radii=4.0, potential_wall_height_hartree=10.0)
    # V = init_zero_potential(N)
    potential_operator = init_potential_operator(V=V, N=N, delta_time=delta_time_h_per_hartree)
    save_potential_image(V=V, volume_width=simulated_volume_width_bohr_radii)

    print("***************************************************************************************")
    print("Starting simulation")

    # Run simulation
    for i in range(1000):
        print("Iteration: ", i, ".")
        wave_tensor = time_evolution(wave_tensor= wave_tensor, kinetic_operator=kinetic_operator, potential_operator=potential_operator)
        probability_density = np.square(np.abs(wave_tensor))
        print(f"Integral of probability density P = {np.sum(probability_density)}.")
        save_probability_density_image(probability_density=probability_density, volume_width=simulated_volume_width_bohr_radii, i=i)

    print("Simulation has finished.")


sim()