import imageio
from sources import volume_visualization


class AnimationWriter:
    def __init__(self, save_path):
        self.writer = imageio.get_writer(save_path)

    def add_frame(self, canvas : volume_visualization.VolumeCanvas):
        im = canvas.canvas.render(alpha=False)
        self.writer.append_data(im)

    def finish(self):
        self.writer.close()