import numpy as np
from numba.experimental import jitclass
from numba import types, typed, jit, njit
import numba


@jitclass(
    [
        ("plane_dwell_time_density", numba.typeof(np.zeros(shape=(256, 256)))),
        ("plane_probability_density", numba.typeof(np.zeros(shape=(256, 256)))),
        ("cumulated_time", numba.typeof(0.0)),
        ("x", numba.typeof(0)),
    ]
)
class MeasurementPlane:
    def __init__(self, wave_tensor, delta_x, location_bohr_radii, simulated_box_width):
        self.plane_dwell_time_density = np.zeros(
            shape=(wave_tensor.shape[0], wave_tensor.shape[1])
        )
        self.plane_probability_density = np.zeros(
            shape=(wave_tensor.shape[0], wave_tensor.shape[1])
        )
        self.cumulated_time = 0.0
        self.x = int((location_bohr_radii + simulated_box_width * 0.5) / delta_x)

    def integrate(self, wave_tensor, delta_time):
        wave_slice = wave_tensor[self.x, :, :]
        self.plane_probability_density = np.square(np.abs(wave_slice))
        self.plane_dwell_time_density += self.plane_probability_density * delta_time
        self.cumulated_time += delta_time

    def get_probability_density(self):
        return self.plane_probability_density

    def get_dwell_time(self):
        return self.plane_dwell_time_density


@jitclass(
    [
        ("bottom_corner", numba.typeof((0, 0, 0))),
        ("top_corner", numba.typeof((0, 0, 0))),
        ("label", numba.typeof("")),
        ("probability", types.float64),
        ("probability_evolution", types.float64[:]),
    ]
)
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
    N: int
    sum_axis: tuple
    label: str

    def __init__(self, N, sum_axis, label):
        self.N = N
        self.sum_axis = sum_axis
        self.probability_density = np.zeros(shape=(N), dtype=np.float64)
        self.label = label

    def integrate_probability_density(self, probability_density_tensor):
        self.probability_density = np.sum(
            a=probability_density_tensor, axis=self.sum_axis
        )

    def get_probability_density_with_label(self):
        return self.probability_density, self.label
