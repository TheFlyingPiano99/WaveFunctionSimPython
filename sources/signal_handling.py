import signal
import sys
from sources.sim_state import SimState
import sources.snapshot_io as snapshot


def register_signal_handler(iter_data):
    def signal_handler(sig, frame):
        iter_data.is_quit = True
        print("Simulation interrupted by the user.")

    signal.signal(signal.SIGINT, signal_handler)
