import imageio
from sources import volume_visualization


class AnimationWriter:
    def __init__(self, save_path):
        self.writer = imageio.get_writer(save_path)

    def add_frame(self, img):
        self.writer.append_data(img)

    def finish(self):
        self.writer.close()
