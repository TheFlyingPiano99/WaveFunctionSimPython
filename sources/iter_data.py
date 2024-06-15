from typing import Dict
from sources.sim_state import SimState
from sources.config_read_helper import try_read_param

class IterData:
    i = 0
    elapsed_system_time_s = 0.0
    average_iteration_system_time_s = 0.0
    total_iteration_count: int
    total_simulated_time = 0.0
    is_quit = False

    def __init__(self, config: Dict):
        self.total_iteration_count = try_read_param(config, "simulation.total_iteration_count", 1000)
