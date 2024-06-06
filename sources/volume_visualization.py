from itertools import cycle

import numpy as np

from vispy import app, scene, io
from vispy.color import get_colormaps, BaseColormap
from vispy.visuals.transforms import STTransform
import vispy.visuals as visuals
import vispy.io as io
import sources.math_utils as math_utils
import sources.multi_volume_visual as multi_volume_visual
import os
import cupy as cp
import warnings
warnings.filterwarnings('ignore')

class VolumetricVisualization:
    canvas: scene.SceneCanvas
    viewing_window_bottom_corner_voxel_3: np.array
    viewing_window_top_corner_voxel_3: np.array
    cam_rotation_speed = 0.0
    cam_elevation_speed = 0.1
    light_rotation_speed = 0.1
    light_elevation_speed = 0.0
    elevation = 45.0
    azimuth = 0.0
    light_elevation_axis: np.array
    multi_volume_visual: multi_volume_visual.MultiVolume

    def __init__(
        self, wave_function: cp.ndarray, potential: cp.ndarray, coulomb_potential: cp.ndarray, cam_rotation_speed=0.0, azimuth=0.0
    ):
        print(wave_function.shape)

        # Prepare canvas
        self.canvas = scene.SceneCanvas(
            keys="interactive", bgcolor="white", size=(1024, 768), show=False
        )
        self.view = self.canvas.central_widget.add_view()

        # create colormaps that work well for translucent density visualisation
        class WaveFunctionDensityColorMap(BaseColormap):
            glsl_map = """
            vec4 translucent_fire(float re, float im) {
                float p = re * re + im * im;
                                
                float phase = atan(im, re) / 3.14159 * 0.5 + 0.5;
                float bias = 0.01;
                float scale = 1.0;
                float tScaled = min((p - bias) * scale, 1.0);
                if (tScaled <= 0.0) {
                    return vec4(0,0,0,0);
                }
                // Mix colors:
                vec3 c0 = vec3(10, 10, 255) / 255.0;
                vec3 c1 = vec3(10, 255, 10) / 255.0;
                vec3 c2 = vec3(255, 10, 10) / 255.0;
                vec3 c3 = vec3(255, 255, 5) / 255.0;
                vec3 color = vec3(0.0);
                if (phase < 0.25){
                    color = c0 * (1.0 - phase / 0.25) + c1 * phase / 0.25;
                }
                else if (phase < 0.5) {
                    color = c1 * (1.0 - (phase - 0.25) / 0.25) + c2 * (phase - 0.25) / 0.25;
                }
                else if (phase < 0.75) {
                    color = c2 * (1.0 - (phase - 0.5) / 0.25) + c3 * (phase - 0.5) / 0.25;
                }
                else {
                    color = c3 * (1.0 - (phase - 0.75) / 0.25) + c0 * (phase - 0.75) / 0.25;
                }
                
                float definition = 1.0;
                return vec4(color, pow(tScaled, definition));
            }
            """

        class PotentialColorMap(BaseColormap):
            glsl_map = """
            vec4 translucent_green(float re, float im) {
                float bias = 0.001;
                float tScaled = min(re - bias, 1.0);
                if (tScaled <= 0.0) {
                    return vec4(0,0,0,0);
                }
                return vec4(tScaled, pow(tScaled, 0.5), tScaled*tScaled, max(0, tScaled*1.001 - 0.001) * 0.5);
            }
            """

        class CoulombColorMap(BaseColormap):
            glsl_map = """
            vec4 translucent_blue(float re, float im) {
                float tScaled = min(re, 1.0);
                return vec4(0, 0, 0, 0);
                //return vec4(tScaled, tScaled*tScaled, pow(tScaled, 0.5), max(0, tScaled*1.001 - 0.001) * 0.01);
            }
            """

        self.wave_function_color_map = WaveFunctionDensityColorMap()
        self.potential_color_map = PotentialColorMap()
        self.coulomb_color_map = CoulombColorMap()

        np_wave_function = cp.asnumpy(wave_function).astype(np.csingle).view(dtype=np.float32).reshape(wave_function.shape + (2,))
        np_potential = cp.asnumpy(potential).astype(np.csingle).view(dtype=np.float32).reshape(potential.shape + (2,))
        np_coulomb = cp.asnumpy(coulomb_potential).astype(np.csingle).view(dtype=np.float32).reshape(coulomb_potential.shape + (2,))

        scale = 0.005
        self.normalized_complex_limit = (
            -1.0 * scale,
            1.0 * scale,
        )

        min_max = max(abs(np_potential.max()), abs(np_potential.min()))
        self.potential_limit = (
            -1.0 * min_max,
            1.0 * min_max,
        )

        npad = ((1,1), (1,1), (1,1), (0,0))
        padded_prob = np.pad(
                    array=np_wave_function,
                    pad_width=npad,
                    mode="constant",
                    constant_values=0.0,
                )
        volumes = [
            (
                padded_prob,
                self.normalized_complex_limit,
                self.wave_function_color_map,
            ),
            (
                np.pad(
                    array=np_potential,
                    pad_width=npad,
                    mode="constant",
                    constant_values=0.0,
                ),
                self.potential_limit,
                self.potential_color_map,
            ),
            (
                np.pad(
                    array=np_coulomb,
                    pad_width=npad,
                    mode="constant",
                    constant_values=0.0,
                ),
                self.potential_limit,
                self.coulomb_color_map,
            ),
        ]
        # Create the volume visuals
        self.multi_volume_visual = multi_volume_visual.MultiVolume(
            volumes=volumes,
            parent=self.view.scene,
            method="translucent",
            relative_step_size=0.1,
        )

        # self.volume.parent=self.view.scene
        # self.volume.method='translucent'
        # self.volume.gamma=1.0
        # self.secondary_volume = scene.visuals.Volume(secondary_data.astype(np.float32), parent=self.view.scene, method='translucent', gamma=1.0)

        self.cam_rotation_speed = cam_rotation_speed
        fov = 45.0
        cam = scene.cameras.TurntableCamera(
            parent=self.view.scene, fov=fov, name="Turntable"
        )
        cam.up = "+y"
        self.azimuth = azimuth
        cam.azimuth = azimuth
        cam.update()
        self.view.camera = cam  # Select turntable at first

        self.text1 = scene.visuals.Text(
            f"Elapsed time = {0.0} ħ/Hartree",
            parent=self.canvas.scene,
            color="black",
        )
        self.text1.font_size = 36
        self.text1.pos = self.canvas.size[0] // 2, self.canvas.size[1] // 14
        self.text1.text = f"Elapsed time = {0.0:.2f} ħ/Hartree = {math_utils.h_bar_per_hartree_to_fs(0.0):.2f} fs"

        # Create Axis:
        # TODO

        """
        # Implement axis connection with cam2
        @self.canvas.events.mouse_move.connect
        def on_mouse_move(event):
            if event.button == 1 and event.is_dragging:
                axis.transform.reset()

                axis.transform.rotate(cam2.roll, (0, 0, 1))
                axis.transform.rotate(cam2.elevation, (1, 0, 0))
                axis.transform.rotate(cam2.azimuth, (0, 1, 0))

                axis.transform.scale((50, 50, 0.001))
                axis.transform.translate((50., 50.))
                axis.update()

        """

    def update(self, wave_function, potential, iter_count, delta_time_h_bar_per_hartree):
        print(wave_function.shape)
        npad = ((1,1), (1,1), (1,1), (0,0))
        self.multi_volume_visual.update_volume_data(
            volume_data=np.pad(
                array=cp.asnumpy(wave_function).astype(np.csingle).view(dtype=np.float32).reshape(wave_function.shape + (2,)),
                pad_width=npad,
                mode="constant",
                constant_values=0.0,
            ),
            index=0,
        )
        self.multi_volume_visual.update_volume_data(
            volume_data=np.pad(
                array=cp.asnumpy(potential).astype(np.csingle).view(dtype=np.float32).reshape(potential.shape + (2,)),
                pad_width=npad,
                mode="constant",
                constant_values=0.0,
            ),
            index=1,
        )

        self.view.camera.azimuth = (
            self.azimuth
            + self.cam_rotation_speed * iter_count * delta_time_h_bar_per_hartree
        )
        self.view.camera.elevation = (
            self.elevation
            + self.cam_elevation_speed * iter_count * delta_time_h_bar_per_hartree
        )
        if self.light_rotation_speed > 0.0:
            self.multi_volume_visual.light_direction = (
                math_utils.rotate(self.multi_volume_visual.light_direction, math_utils.prefered_up(), delta_time_h_bar_per_hartree * self.light_rotation_speed)
            )

        if self.light_elevation_speed != 0.0:
            self.multi_volume_visual.light_direction = (
                math_utils.rotate(self.multi_volume_visual.light_direction, np.array([1, 0, 0], dtype=np.float_), delta_time_h_bar_per_hartree * self.light_elevation_speed)
            )

        self.canvas.update()
        self.text1.text = f"Elapsed time = {iter_count * delta_time_h_bar_per_hartree:.2f} ħ/Hartree = {math_utils.h_bar_per_hartree_to_fs(iter_count * delta_time_h_bar_per_hartree):.2f} fs"
        return self.canvas

    def render_to_png(self, out_dir, index):
        img = self.render()
        dir = os.path.join(out_dir, "probability_density_3D/")
        file_name = f"probability_density_3D_{index:04d}.png"
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)
        io.write_png(filename=os.path.join(dir, file_name), data=img)

    def get_canvas(self):
        return self.canvas

    def render(self):
        return self.canvas.render(alpha=False)

    def set_light_direction(self, light_dir: np.array):
        self.multi_volume_visual.light_direction = light_dir
