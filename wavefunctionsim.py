from sources import volumetric_visualization, animation, measurement, text_writer
import sources.initialisation as init
import sources.core_sim as core_sim
import sys
import os
import signal
import sources.text_writer as text_writer
from colorama import Fore, Style


def parse_arguments():
    arguments = sys.argv
    arguments.pop(0)
    use_cache = True
    is_help = False
    is_version = False
    is_error = False
    if "-nc" in arguments:
        arguments.remove("-nc")
        use_cache = False
    if "--help" in arguments or "-h" in arguments:
        try:
            arguments.remove("--help")
        except ValueError:
            pass
        try:
            arguments.remove("-h")
        except ValueError:
            pass
        is_help = True
    if "--version" in arguments:
        arguments.remove("--version")
        is_version = True

    if len(arguments) > 0:  # Exit if received unknown parameter
        print(f"Unknown argument: {arguments}")
        print("To see the list of available arguments use \"-h\" or \"--help\" !")
        is_error = True
    return use_cache, is_help, is_version, is_error

def sim():
    print(text_writer.get_title_text() + "\n")

    use_cache, is_help, is_version, is_error = parse_arguments()

    if is_error:
        return

    if is_help:
        print(text_writer.get_help_text())
        return

    if is_version:
        print(text_writer.get_version_text())
        return

    # Check display availability:
    have_display = "DISPLAY" in os.environ
    if not have_display:
        exitval = os.system('python -c "import matplotlib.pyplot as plt; plt.figure()"')
        have_display = (exitval == 0)
    if have_display:
        print("Display connected.\n")
    else:
        print("Display not connected.\n")

    if not have_display:    # Create virtual display if no physical connected
        from xvfbwrapper import Xvfb
        vdisplay = Xvfb()
        vdisplay.start()
        print("Created virtual display.\n")

    os.environ["CUPY_ACCELERATORS"] = "cub.docs.cupy.dev/en/stable/reference/environment.html"

    # Initialization:
    sim_state, measurement_tools, iter_data = init.initialize(use_cache)

    text_writer.write_sim_state(sim_state)

    print(
        "\n****************************************************************************\n"
    )

    # Simulation:
    sim_state, measurement_tools, iter_data = core_sim.run_iteration(
        sim_state, measurement_tools, iter_data
    )

    # Finishing steps:
    measurement_tools.finish(sim_state)
    print(Fore.GREEN + "Simulation has finished." + Style.RESET_ALL)
    print(text_writer.get_finish_text(iter_data))
    text_writer.append_iter_data(iter_data, sim_state)

    if not have_display and sim_state.enable_visual_output:
        vdisplay.stop()


if __name__ == "__main__":
    sim()
