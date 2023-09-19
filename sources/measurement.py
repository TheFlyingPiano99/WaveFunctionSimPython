import numpy as np
import matplotlib.image

class MeasurementPlane:

    def __init__(self, wave_tensor, delta_x, location_bohr_radii):
        self.plane_dwell_time_density = np.zeros(shape=(wave_tensor.shape[0], wave_tensor.shape[1]))
        self.plane_probability_density = np.zeros(shape=(wave_tensor.shape[0], wave_tensor.shape[1]))
        self.cumulated_time = 0.0
        self.x = int(location_bohr_radii / delta_x)

    def integrate(self, wave_tensor, delta_time):
        wave_slice = wave_tensor[:, :, self.x]
        self.plane_probability_density = np.square(np.abs(wave_slice))
        self.plane_dwell_time_density += self.plane_probability_density * delta_time
        self.cumulated_time += delta_time


    def save(self, probability_save_path, dwell_time_save_path):
        formatted = self.plane_probability_density
        matplotlib.image.imsave(fname=probability_save_path, arr=formatted, cmap='gist_heat', dpi=100)
        formatted = self.plane_dwell_time_density
        matplotlib.image.imsave(fname=dwell_time_save_path, arr=formatted, cmap='gist_heat', dpi=100)

class AAMeasurementVolume:

    def __init__(self, bottom_corner, top_corner, label='Unknown'):
        self.top_corner = top_corner
        self.bottom_corner =bottom_corner
        self.label = label
        self.probability = 0.0
        self.probability_evolution = np.empty(shape=0, dtype=np.float64)


    def integrate_probability_density(self, probability_density):
        self.probability = np.sum(probability_density[self.bottom_corner[0]:self.top_corner[0], self.bottom_corner[1]:self.top_corner[1], self.bottom_corner[2]:self.top_corner[2]])
        self.probability_evolution = np.append(arr=self.probability_evolution, values=self.probability)

    def get_probability(self):
        return self.probability

    def get_probability_evolution(self):
        return self.probability_evolution, self.label