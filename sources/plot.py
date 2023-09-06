import plotly.graph_objects as go
import numpy as np
import math

def plot_potential_image(V, N, delta_x):
    x, y, z = np.mgrid[0:N, 0:N, 0:N]
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
            xaxis=dict(title=f'x [{delta_x:.2f} Bohr radii]'),
            yaxis=dict(title=f'y [{delta_x:.2f} Bohr radii]'),
            zaxis=dict(title=f'z [{delta_x:.2f} Bohr radii]'),
        ),
        scene_camera_eye=dict(
            x=2.0 * math.sin(0),
            y=2.0 * math.cos(0),
            z=1.0
        )
    )

    # Show the figure
    fig.write_image(f"images/potential_energy.jpeg")


def plot_probability_density_image(probability_density, delta_time_h_per_hartree, delta_x, N, i : int):
    x, y, z = np.mgrid[0:N, 0:N, 0:N]
    fig = go.Figure()
    fig.add_trace(go.Volume(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value = probability_density.flatten(),
        isomin=0.0,
        isomax= 20 / (N * N * N),
        opacity=0.3,
        surface_count=25,
        colorscale='Viridis',
        colorbar_title='Probability density'
    ))

    # Set the layout of the figure
    fig.update_layout(
        title="Probability density ({}. time step; Elapsed time = {:.2f} h-bar / hartree)".format(i, delta_time_h_per_hartree * i),
        scene=dict(
            xaxis=dict(title=f'x [{delta_x:.2f} Bohr radii]'),
            yaxis=dict(title=f'y [{delta_x:.2f} Bohr radii]'),
            zaxis=dict(title=f'z [{delta_x:.2f} Bohr radii]'),

        ),
        scene_camera_eye=dict(
            x=2.0 * math.sin(i / 200.0 * math.pi),
            y=2.0 * math.cos(i / 200.0 * math.pi),
            z=1.0
        )
    )

    # Show the figure
    fig.write_image(f"images/probability_density{i:03d}.jpeg")