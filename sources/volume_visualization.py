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


class VolumetricVisualization:
    canvas: scene.SceneCanvas
    viewing_window_bottom_corner_voxel: np.array
    viewing_window_top_corner_voxel: np.array
    density_scale = 100000.0
    cam_rotation_speed = 0.0
    cam_elevation_speed = 0.1
    light_rotation_speed = 0.1
    light_elevation_speed = 0.0
    elevation = 45.0
    azimuth = 0.0
    light_elevation_axis: np.array

    def __init__(
        self, volume_data: np.ndarray, secondary_volume_data: np.ndarray, cam_rotation_speed=0.0, azimuth=0.0
    ):
        volume_data = volume_data * self.density_scale
        # Prepare canvas
        self.canvas = scene.SceneCanvas(
            keys="interactive", bgcolor="black", size=(1024, 768), show=False
        )
        self.view = self.canvas.central_widget.add_view()

        # create colormaps that work well for translucent density visualisation
        class ProbabilityDensityColorMap(BaseColormap):
            glsl_map = """
            vec4 translucent_fire(float t) {
                float tScaled = min(t, 1.0);
                return vec4(pow(tScaled, 0.1), pow(tScaled, 0.9), pow(tScaled, 2.0), pow(tScaled, 1.0));
            }
            """

        class PotentialColorMap(BaseColormap):
            glsl_map = """
            vec4 translucent_green(float t) {
                float tScaled = min(t, 1.0);
                return vec4(tScaled, pow(tScaled, 0.5), tScaled*tScaled, max(0, tScaled*1.001 - 0.001) * 0.5);
            }
            """

        self.primary_color_map = ProbabilityDensityColorMap()
        self.secondary_color_map = PotentialColorMap()
        self.clim1 = (
            0.0,
            volume_data.astype(np.float32).max() * 0.01,
        )
        self.clim2 = (
            secondary_volume_data.astype(np.float32).min(),
            secondary_volume_data.astype(np.float32).max(),
        )

        volumes = [
            (
                np.pad(
                    array=volume_data.astype(np.float32),
                    pad_width=1,
                    mode="constant",
                    constant_values=0.0,
                ),
                self.clim1,
                self.primary_color_map,
            ),
            (
                np.pad(
                    array=secondary_volume_data.astype(np.float32),
                    pad_width=1,
                    mode="constant",
                    constant_values=0.0,
                ),
                self.clim2,
                self.secondary_color_map,
            ),
        ]
        # Create the volume visuals
        self.volume = multi_volume_visual.MultiVolume(
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
            f"Probability density (Elapsed time = {0.0} ħ/E)",
            parent=self.canvas.scene,
            color="white",
        )
        self.text1.font_size = 16
        self.text1.pos = self.canvas.size[0] // 2, self.canvas.size[1] // 14
        self.text1.text = f"Probability density (Elapsed time = {0.0:.5f} ħ/E = {math_utils.h_bar_per_hartree_to_ns(0.0):.2E} ns)"

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

    def update(self, volume_data, iter_count, delta_time_h_bar_per_hartree):
        self.volume.update_volume_data(
            volume_data=np.pad(
                array=(volume_data * self.density_scale).astype(np.float32),
                pad_width=1,
                mode="constant",
                constant_values=0.0,
            ),
            index=0,
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
            self.volume.light_direction = (
                math_utils.rotate(self.volume.light_direction, math_utils.prefered_up(), delta_time_h_bar_per_hartree * self.light_rotation_speed)
            )

        if self.light_elevation_speed != 0.0:
            self.volume.light_direction = (
                math_utils.rotate(self.volume.light_direction, np.array([1,0,0], dtype=np.float_), delta_time_h_bar_per_hartree * self.light_elevation_speed)
            )

        self.canvas.update()
        self.text1.text = f"Probability density (Elapsed time = {iter_count * delta_time_h_bar_per_hartree:.5f} ħ/E = {math_utils.h_bar_per_hartree_to_ns(iter_count * delta_time_h_bar_per_hartree):.2E} ns)"
        return self.canvas

    def render_to_png(self, index):
        img = self.render()
        dir = "output/probability_density_3D/"
        file_name = f"probability_density_3D_{index:04d}.png"
        if not os.path.exists(dir):
            os.mkdir(dir)
        io.write_png(filename=os.path.join(dir, file_name), data=img)

    def get_canvas(self):
        return self.canvas

    def render(self):
        return self.canvas.render(alpha=False)

    def set_light_direction(self, light_dir: np.array):
        self.volume.light_direction = light_dir
