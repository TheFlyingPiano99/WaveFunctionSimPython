import plotly.graph_objects as go
import numpy as np
import math

def plot_potential_image(V, volume_width):
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


def plot_probability_density_image(probability_density, volume_width, i : int):
    x, y, z = np.mgrid[:volume_width, :volume_width, :volume_width]
    fig = go.Figure()
    fig.add_trace(go.Volume(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value = probability_density.flatten(),
        isomin=0.0,
        isomax= 1.0 / (volume_width * volume_width * volume_width),
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