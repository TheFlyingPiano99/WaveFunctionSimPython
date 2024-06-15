import imageio


class AnimationWriter:
    video_frame_rate: int
    frame_capture_iteration_interval: int

    def __init__(self, save_path, video_frame_rate: int = 25, capture_interval: int = 1):
        self.video_frame_rate = video_frame_rate
        self.frame_capture_iteration_interval = capture_interval
        self.writer = imageio.get_writer(save_path, fps=video_frame_rate)

    def add_frame(self, img):
        self.writer.append_data(img)

    def finish(self):
        self.writer.close()
