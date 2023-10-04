import signal
import sys
from sources.sim_state import SimState
import sources.snapshot_io as snapshot


def register_signal_handler(sim_state: SimState, iter_data):
    def signal_handler(sig, frame):
        print("Simulation interrupted by the user.")
        snapshot.write_snapshot(sim_state, iter_data)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
