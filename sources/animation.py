import imageio


class AnimationWriter:
    def __init__(self, save_path):
        self.writer = imageio.get_writer(save_path, fps=20)

    def add_frame(self, img):
        self.writer.append_data(img)

    def finish(self):
        self.writer.close()
