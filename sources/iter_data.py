from typing import Dict
from sources.sim_state import SimState
from sources.config_read_helper import try_read_param

class IterData:
    i = 0
    elapsed_system_time_s = 0.0
    average_iteration_system_time_s = 0.0
    animation_frame_step_interval: int
    png_step_interval: int
    measurement_plane_capture_interval: int
    probability_plot_interval: int
    total_iteration_count: int
    total_simulated_time = 0.0
    per_axis_probability_denisty_plot_interval: int
    wave_function_save_interval: int
    is_quit = False

    def __init__(self, config: Dict):
        self.total_iteration_count = try_read_param(config, "simulation.total_iteration_count", 1000)
