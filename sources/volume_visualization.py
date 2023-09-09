from itertools import cycle

import numpy as np

from vispy import app, scene, io
from vispy.color import get_colormaps, BaseColormap
from vispy.visuals.transforms import STTransform
import vispy.io as io


class VolumeCanvas:

    def __init__(self, volume_data, secondary_data):
        # Prepare canvas
        self.canvas = scene.SceneCanvas(keys='interactive', bgcolor='black', size=(800, 600), show=False)

        # Set up a viewbox to display the image with interactive pan/zoom
        self.view = self.canvas.central_widget.add_view()

        # Create the volume visuals
        self.volume = scene.visuals.Volume(volume_data.astype(np.float32), parent=self.view.scene, method='translucent', gamma=1.0)
        self.secondary_volume = scene.visuals.Volume(secondary_data.astype(np.float32), parent=self.view.scene, method='translucent', gamma=1.0)

        fov = 60.
        cam = scene.cameras.TurntableCamera(parent=self.view.scene, fov=fov,
                                             name='Turntable')
        self.view.camera = cam  # Select turntable at first

        # Create an XYZAxis visual
        axis = scene.visuals.XYZAxis(parent=self.view)
        s = STTransform(translate=(50, 50), scale=(50, 50, 50, 1))
        affine = s.as_matrix()
        axis.transform = affine

        # Create Axis:
        # TODO

        # create colormaps that work well for translucent density visualisation
        class CustomColorMap(BaseColormap):
            glsl_map = """
            vec4 translucent_fire(float t) {
                float tScaled = min(t * 50.0, 1.0);
                return vec4(pow(tScaled, 0.5), tScaled, tScaled*tScaled, max(0, tScaled*1.05 - 0.05));
            }
            """
        class SecundaryColorMap(BaseColormap):
            glsl_map = """
            vec4 translucent_fire(float t) {
                float tScaled = min(t * 0.1, 1.0);
                return vec4(tScaled, pow(tScaled, 0.5), tScaled*tScaled, max(0, tScaled*1.05 - 0.05));
            }
            """
        self.volume.cmap = CustomColorMap()
        self.secondary_volume.cmap = SecundaryColorMap()

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

        # for testing performance
        # @canvas.connect
        # def on_draw(ev):
        # canvas.update()

    def update(self, volume_data):
        self.volume.set_data(volume_data)
        self.volume.update()
        self.canvas.update()
        return self.canvas

    def save_to_png(self, save_path):
        img = self.canvas.render()
        io.write_png(filename=save_path, data=img)

    def get_canvas(self):
        return self.canvas