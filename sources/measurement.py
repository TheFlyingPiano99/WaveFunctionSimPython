import numpy as np
from numba.experimental import jitclass
from numba import types
import numba


class MeasurementPlane:
    def __init__(
        self,
        wave_tensor,
        delta_x,
        location_bohr_radii,
        simulated_box_width,
        viewing_window_bottom_voxel,
        viewing_window_top_voxel,
    ):
        self.plane_dwell_time_density = np.zeros(
            shape=(
                viewing_window_top_voxel[2] - viewing_window_bottom_voxel[2],
                viewing_window_top_voxel[1] - viewing_window_bottom_voxel[1],
            )
        )
        self.plane_probability_density = np.zeros(
            shape=(
                viewing_window_top_voxel[2] - viewing_window_bottom_voxel[2],
                viewing_window_top_voxel[1] - viewing_window_bottom_voxel[1],
            )
        )
        self.cumulated_time = 0.0
        self.x = int((location_bohr_radii + simulated_box_width * 0.5) / delta_x)
        self.viewing_window_bottom_voxel = viewing_window_bottom_voxel
        self.viewing_window_top_voxel = viewing_window_top_voxel

    def integrate(self, probability_density, delta_time):
        self.plane_probability_density = probability_density[
            self.x,
            self.viewing_window_bottom_voxel[1] : self.viewing_window_top_voxel[1],
            self.viewing_window_bottom_voxel[2] : self.viewing_window_top_voxel[2],
        ]
        self.plane_dwell_time_density += self.plane_probability_density * delta_time
        self.cumulated_time += delta_time

    def get_probability_density(self):
        return self.plane_probability_density

    def get_dwell_time(self):
        return self.plane_dwell_time_density


class AAMeasurementVolume:
    def __init__(
        self,
        bottom_corner,
        top_corner,
        label,
    ):
        self.top_corner = top_corner
        self.bottom_corner = bottom_corner
        self.label = label
        self.probability = 0.0
        self.probability_evolution = np.empty(shape=0, dtype=np.float64)

    def integrate_probability_density(self, probability_density):
        self.probability = np.sum(
            probability_density[
                self.bottom_corner[0] : self.top_corner[0],
                self.bottom_corner[1] : self.top_corner[1],
                self.bottom_corner[2] : self.top_corner[2],
            ]
        )
        self.probability_evolution = np.append(
            arr=self.probability_evolution, values=self.probability
        )

    def get_probability(self):
        return self.probability

    def get_probability_evolution(self):
        return self.probability_evolution, self.label


"""
For calculating the probability density per axis
without any assuptions about the other dimensions.
"""


class ProjectedMeasurement:
    probability_density: np.array
    min_voxel: int
    max_voxel: int
    left_edge_bohr_radii: float
    right_edge_boh_radii: float
    sum_axis: tuple
    label: str

    def __init__(self, min_voxel, max_voxel, left_edge, right_edge, sum_axis, label):
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

    def integrate_probability_density(self, probability_density_tensor):
        self.probability_density = np.sum(
            a=probability_density_tensor, axis=self.sum_axis
        )[self.min_voxel : self.max_voxel]

    def get_probability_density_with_label(self):
        return (
            self.probability_density,
            self.label,
            self.left_edge_bohr_radii,
            self.right_edge_bohr_radii,
        )
