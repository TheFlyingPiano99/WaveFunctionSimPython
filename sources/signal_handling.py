import signal
import sys
from sources.sim_state import SimState
import sources.snapshot_io as snapshot
from colorama import Fore, Style


def register_signal_handler(iter_data):
    def signal_handler(sig, frame):
        if not iter_data.is_quit:
            print(Fore.RED + "Simulation interrupted by the user." + Style.RESET_ALL)
        iter_data.is_quit = True

    signal.signal(signal.SIGINT, signal_handler)
