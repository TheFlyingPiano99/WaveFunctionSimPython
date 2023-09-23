from itertools import cycle

import numpy as np

from vispy import app, scene, io
from vispy.color import get_colormaps, BaseColormap
from vispy.visuals.transforms import STTransform
import vispy.visuals as visuals
import vispy.io as io
import sources.math_utils as math_utils
import sources.multi_volume_visual as multi_volume_visual

class VolumeCanvas:

    def __init__(self, volume_data, secondary_data):
        # Prepare canvas
        self.canvas = scene.SceneCanvas(keys='interactive', bgcolor='black', size=(1024, 768), show=False)
        self.view = self.canvas.central_widget.add_view()

        """
        # Create an XYZAxis visual
        axis = scene.visuals.XYZAxis(parent=self.view.scene)
        s = STTransform(translate=(-10, -10, -10), scale=(volume_data.shape[0], volume_data.shape[1], volume_data.shape[2], 1))
        affine = s.as_matrix()
        axis.transform = affine
        axis.update()
        """

        # create colormaps that work well for translucent density visualisation
        class ProbabilityDensityColorMap(BaseColormap):
            glsl_map = """
            vec4 translucent_fire(float t) {
                float tScaled = min(t * 50.0, 1.0);
                return vec4(pow(tScaled, 0.1), pow(tScaled, 0.9), pow(tScaled, 2.0), max(0, tScaled*1.05 - 0.05));
            }
            """
        class PotentialColorMap(BaseColormap):
            glsl_map = """
            vec4 translucent_green(float t) {
                float tScaled = min(t * 1.0, 1.0);
                return vec4(tScaled, pow(tScaled, 0.5), tScaled*tScaled, max(0, tScaled*1.05 - 0.05) * 0.1);
            }
            """
        self.primary_color_map = ProbabilityDensityColorMap()
        self.secondary_color_map = PotentialColorMap()
        self.clim1 = volume_data.astype(np.float32).min(), volume_data.astype(np.float32).max()
        self.clim2 = secondary_data.astype(np.float32).min(), secondary_data.astype(np.float32).max()

        # Create the volume visuals
        self.volume = multi_volume_visual.MultiVolume(volumes=[(volume_data.astype(np.float32), self.clim1, self.primary_color_map),
                                                                     (secondary_data.astype(np.float32), self.clim2, self.secondary_color_map)
                                                                     ],
                                                            parent=self.view.scene, method='translucent')

        #self.volume.parent=self.view.scene
        #self.volume.method='translucent'
        #self.volume.gamma=1.0
        #self.secondary_volume = scene.visuals.Volume(secondary_data.astype(np.float32), parent=self.view.scene, method='translucent', gamma=1.0)

        fov = 60.
        cam = scene.cameras.TurntableCamera(parent=self.view.scene, fov=fov,
                                             name='Turntable')
        cam.update()
        self.view.camera = cam  # Select turntable at first

        self.text1 = scene.visuals.Text(f'Probability density (Elapsed time = {0.0} ħ/E)', parent=self.canvas.scene, color='white')
        self.text1.font_size = 16
        self.text1.pos = self.canvas.size[0] // 2, self.canvas.size[1] // 14
        self.text1.text = f'Probability density (Elapsed time = {0.0:.5f} ħ/E = {math_utils.h_bar_per_hartree_to_ns(0.0):.2E} ns)'

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
        self.volume.update_volume_data(volume_data=volume_data.astype(np.float32), index=0)
        self.canvas.update()
        self.text1.text = f'Probability density (Elapsed time = {iter_count * delta_time_h_bar_per_hartree:.5f} ħ/E = {math_utils.h_bar_per_hartree_to_ns(iter_count * delta_time_h_bar_per_hartree):.2E} ns)'
        return self.canvas

    def save_to_png(self, save_path):
        img = self.canvas.render()
        io.write_png(filename=save_path, data=img)

    def get_canvas(self):
        return self.canvas