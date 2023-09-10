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
