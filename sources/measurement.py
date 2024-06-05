import numpy as np
from numba.experimental import jitclass
from numba import types
import numba
import sources.volume_visualization as volume_visualization
import sources.animation as animation

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
        bottom_corner : np.array,
        top_corner: np.array,
        label : str,
    ):
        self.top_corner = top_corner
        self.bottom_corner = bottom_corner
        self.label = label
        self.probability = 0.0
        self.probability_evolution = np.empty(shape=0, dtype=np.float64)

    def integrate_probability_density(self, probability_density):
        self.probability = min(np.sum(
            probability_density[
                self.bottom_corner[0] : self.top_corner[0],
                self.bottom_corner[1] : self.top_corner[1],
                self.bottom_corner[2] : self.top_corner[2],
            ]
        ), 10.0)
        self.probability_evolution = np.append(
            arr=self.probability_evolution, values=self.probability
        )

    def clear(self):
        self.probability = 0.0
        self.probability_evolution = np.empty(shape=0, dtype=np.float64)

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
    near_voxel: int     # in directions of summing axes
    far_voxel: int      # in directions of summing axes
    left_edge_bohr_radii: float
    right_edge_boh_radii: float
    sum_axis: tuple
    label: str
    scale_factor: float = 1.0
    offset: float = 0.0

    def __init__(self,
                 min_voxel : int,
                 max_voxel : int,
                 left_edge : float,
                 right_edge : float,
                 sum_axis : tuple,
                 label : str,
                 near_voxel : int = None,
                 far_voxel : int = None,
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
            self.near_voxel = self.min_voxel     # let's assume cube shaped viewing window
        else:
            self.near_voxel = near_voxel
        if far_voxel is None:
            self.far_voxel = self.max_voxel     # let's assume cube shaped viewing window
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
    measurement_plane: MeasurementPlane
    volumetric: volume_visualization.VolumetricVisualization
    animation_writer_3D: animation.AnimationWriter
    animation_writer_per_axis: animation.AnimationWriter
    x_axis_probability_density: ProjectedMeasurement
    y_axis_probability_density: ProjectedMeasurement
    z_axis_probability_density: ProjectedMeasurement
    projected_potential: ProjectedMeasurement
