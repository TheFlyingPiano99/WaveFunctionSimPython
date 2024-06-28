import os
import sys
import glob
import numpy as np
import toml
import sources.config_read_helper as config_helper
import sources.plot as plot
import sources.math_utils as math_utils
import sources.qt_viewer_app as qt_viewer


def main():
    arguments = sys.argv
    out_folder = "output"
    if len(arguments) > 1 and arguments[1] not in ["-h", "-help", "--help", "-p", "-pred", "--pred"]:
        out_folder = arguments[1]
    f_npy = "measurable_npy"
    f_bin = "measurable_raw"
    f_img = "images"

    is_prediction = False
    # Check flags:
    if "-h" in arguments or "-help" in arguments or "--help" in arguments:
        print(
            "This is a utility program that reads and visualizes the output of the simulator.\n"
            "Usage:\n"
            "python wfreader.py <path to the output folder> [-p]\n"
            "The \"-p\" flag enables the analytic prediction of the expected location and deviation of a Gauss WP in free space."
        )
        return
    if "-p" in arguments or "-pred" in arguments or "--pred" in arguments:
        is_prediction = True
    # Check output folder existence:
    if not os.path.exists(out_folder):
        message = f"No output folder with name {out_folder} found!"
        raise OSError(message)

    # Read saved configuration:
    config = toml.load(os.path.join(out_folder, "config.toml"))
    delta_t = config_helper.try_read_param(config, "simulation.delta_time_h_bar_per_hartree", 1.0)
    position = np.array(config_helper.try_read_param(config, "wave_packet.initial_position_bohr_radii_3", [0.0, 0.0, 0.0]))
    velocity = np.array(config_helper.try_read_param(config, "wave_packet.initial_velocity_bohr_radii_hartree_per_h_bar_3", [0.0, 0.0, 0.0]))
    mass = config_helper.try_read_param(config, "wave_packet.particle_mass_electron_rest_mass", 1.0)
    sigma = np.array(config_helper.try_read_param(config, "wave_packet.initial_standard_deviation_bohr_radii_3", [1.0, 1.0, 1.0]))
    print(f"delta_t = {delta_t} h-bar/Hartree")
    simulated_size_bohr_radii_3 = np.array(config_helper.try_read_param(config, "volume.simulated_volume_dimensions_bohr_radii_3", [120.0, 120.0, 120.0]))
    print(f"Simulated volume size: {simulated_size_bohr_radii_3} Bohr radii")
    voxel_count = np.array(config_helper.try_read_param(config, "volume.number_of_voxels_3", [256, 256, 256]))
    print(f"Voxel count: {voxel_count} voxels")
    delta_r = simulated_size_bohr_radii_3 / voxel_count
    print(f"delta_r = {delta_r} Bohr radii")
    observation_box_bottom_corner_bohr_radii_3 = np.array(config_helper.try_read_param(config, "volume.observation_box_bottom_corner_bohr_radii_3", [-30.0, -30.0, -30.0]))
    print(f"Observation box bottom corner = {observation_box_bottom_corner_bohr_radii_3} Bohr radii")
    observation_box_top_corner_bohr_radii_3 = np.array(config_helper.try_read_param(config, "volume.observation_box_top_corner_bohr_radii_3", [30.0, 30.0, 30.0]))
    print(f"Observation box top corner = {observation_box_top_corner_bohr_radii_3} Bohr radii")
    observation_box_size = observation_box_top_corner_bohr_radii_3 - observation_box_bottom_corner_bohr_radii_3
    figures = []

    # Plot output:
    print("\nExpected location:\n")
    f = os.path.join(out_folder, f_npy, "expected_location_evolution.npy")
    print(f)
    location = np.load(f)
    expected_location_evolution_with_label = list(zip(location.T, ["X axis", "Y axis", "Z axis"]))
    if is_prediction:
        expected_location_evolution_with_label.append(
            [math_utils.predict_free_space_expected_location(
                delta_t=delta_t,
                step_count=expected_location_evolution_with_label[0][0].size,
                velocity=velocity[0],
                x0=position[0],
            ),"X axis (pred.)", "dotted"]
        )
        expected_location_evolution_with_label.append(
            [math_utils.predict_free_space_expected_location(
                delta_t=delta_t,
                step_count=expected_location_evolution_with_label[0][0].size,
                velocity=velocity[1],
                x0=position[1],
            ), "Y axis (pred.)", "dotted"]
        )
        expected_location_evolution_with_label.append(
            [math_utils.predict_free_space_expected_location(
                delta_t=delta_t,
                step_count=expected_location_evolution_with_label[0][0].size,
                velocity=velocity[2],
                x0=position[2],
            ), "Z axis (pred.)", "dotted"]
        )
    figures.append(plot.plot_probability_evolution(
        out_dir=None,
        file_name=None,
        title="Expected location evolution",
        y_label="Expected location [Bohr radius]",
        probability_evolutions=expected_location_evolution_with_label,
        delta_t=delta_t,
        show_fig=False,
        y_min=np.min(observation_box_bottom_corner_bohr_radii_3),
        y_max=np.max(observation_box_top_corner_bohr_radii_3),
    ))

    print("\nStandard deviation:\n")
    f = os.path.join(out_folder, f_npy, "standard_deviation_evolution.npy")
    print(f)
    deviation = np.load(f)
    standard_deviation_with_label = list(zip(deviation.T, ["X axis", "Y axis", "Z axis"]))
    if is_prediction:
        standard_deviation_with_label.append(
            [math_utils.predict_free_space_standard_devation(
                delta_t=delta_t,
                mass=mass,
                step_count=standard_deviation_with_label[0][0].size,
                sigma0=sigma[0],
            ), "X axis (pred.)", "dotted"]
        )
        standard_deviation_with_label.append(
            [math_utils.predict_free_space_standard_devation(
                delta_t=delta_t,
                mass=mass,
                step_count=standard_deviation_with_label[0][0].size,
                sigma0=sigma[1],
            ), "Y axis (pred.)", "dotted"]
        )
        standard_deviation_with_label.append(
            [math_utils.predict_free_space_standard_devation(
                delta_t=delta_t,
                mass=mass,
                step_count=standard_deviation_with_label[0][0].size,
                sigma0=sigma[2],
            ), "Z axis (pred.)", "dotted"]
        )
    figures.append(
        plot.plot_probability_evolution(
            out_dir=None,
            file_name=None,
            title="Standard deviation evolution",
            y_label="Standard deviation [Bohr radius]",
            probability_evolutions=standard_deviation_with_label,
            delta_t=delta_t,
            show_fig=False,
            y_min=0.0,
            y_max=np.max(observation_box_top_corner_bohr_radii_3) * 0.5,
        )
    )

    print("\nProbability currents:\n")
    prob_current_evolutions = []
    for f in glob.glob(os.path.join(out_folder, f_npy, "probability_current_evolution_*.npy")):
        print(f)
        prob_current = np.load(f)
        print(f"Sample count: {prob_current.size}")
        name = f.strip(os.path.join(out_folder, f_npy, "probability_current_evolution_")).strip(".npy")
        prob_current_evolutions.append([prob_current, name])
    figures.append(plot.plot_probability_evolution(
        out_dir=None,
        probability_evolutions=prob_current_evolutions,
        file_name=None,
        delta_t=delta_t,
        title="Probability current evolution",
        y_label="Probability current",
        show_fig=False,
        y_min=-1.0,
        y_max=1.0,
    ))

    print("\nIntegrated probability currents:\n")
    int_prob_current_evolutions = []
    for f in glob.glob(os.path.join(out_folder, f_npy, "integrated_probability_current_*.npy")):
        print(f)
        int_prob_current = np.load(f)
        print(f"Sample count: {int_prob_current.size}")
        name = f.strip(os.path.join(out_folder, f_npy, "integrated_probability_current_")).strip(".npy")
        int_prob_current_evolutions.append([int_prob_current, name])
    figures.append(plot.plot_probability_evolution(
        out_dir=None,
        probability_evolutions=int_prob_current_evolutions,
        file_name=None,
        title="Integrated probability current evolution",
        y_label="Probability",
        delta_t=delta_t,
        show_fig=False,
        y_min=-1.1,
        y_max=1.1,
    ))

    print("\nVolume probabilities:\n")
    prob_evolutions = []
    for f in glob.glob(os.path.join(out_folder, f_npy, "volume_probability_evolution_*.npy")):
        print(f)
        prob = np.load(f)
        print(f"Sample count: {prob.size}")
        name = f.strip(os.path.join(out_folder, f_npy, "volume_probability_evolution_")).strip(".npy")
        prob_evolutions.append([prob, name])
    sum = np.array(
        np.zeros(shape=prob_evolutions[0][0].shape, dtype=prob_evolutions[0][0].dtype).tolist()
    )
    for evolution in prob_evolutions:
        sum = np.add(sum, np.array(evolution[0].tolist()))
    prob_evolutions.append([sum, "Sum"])

    figures.append(plot.plot_probability_evolution(
        out_dir=None,
        probability_evolutions=prob_evolutions,
        file_name=None,
        delta_t=delta_t,
        title="Volume probability evolution",
        y_label="Probability",
        show_fig=False,
        y_min=-0.1,
        y_max=1.1,
    ))

    qt_viewer.start_app(figures)


if __name__ == "__main__":
    main()