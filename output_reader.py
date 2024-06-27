import os
import sys
import glob
import numpy as np
import toml
import sources.config_read_helper as config_helper
import sources.plot as plot


def main():
    arguments = sys.argv
    out_folder = arguments[1]
    if not os.path.exists(out_folder):
        raise f"No output folder with name {out_folder} found!"

    config = toml.load(os.path.join(out_folder, "config.toml"))
    delta_t = config_helper.try_read_param(config, "simulation.delta_time_h_bar_per_hartree", 1.0)
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

    print("\nExpected location:\n")
    f = os.path.join(out_folder, "expected_location_evolution.npy")
    print(f)
    location = np.load(f)
    expected_location_evolution_with_label = list(zip(location.T, ["X axis", "Y axis", "Z axis"]))
    plot.plot_probability_evolution(
        out_dir=None,
        file_name=None,
        title="Expected location evolution",
        y_label="Expected location [Bohr radius]",
        probability_evolutions=expected_location_evolution_with_label,
        delta_t=delta_t,
        show_fig=True,
        y_min=np.min(observation_box_bottom_corner_bohr_radii_3),
        y_max=np.max(observation_box_top_corner_bohr_radii_3),
    )

    print("\nStandard deviation:\n")
    f = os.path.join(out_folder, "standard_deviation_evolution.npy")
    print(f)
    deviation = np.load(f)
    standard_deviation_with_label = list(zip(deviation.T, ["X axis", "Y axis", "Z axis"]))
    plot.plot_probability_evolution(
        out_dir=None,
        file_name=None,
        title="Standard deviation evolution",
        y_label="Standard deviation [Bohr radius]",
        probability_evolutions=standard_deviation_with_label,
        delta_t=delta_t,
        show_fig=True,
        y_min=0.0,
        y_max=np.max(observation_box_top_corner_bohr_radii_3) * 0.5,
    )

    print("\nVolume probabilities:\n")
    prob_evolutions = []
    for f in glob.glob(os.path.join(out_folder, "volume_probability_evolution_*.npy")):
        print(f)
        prob = np.load(f)
        print(f"Sample count: {prob.size}")
        name = f.strip("\\volume_probability_evolution_").strip(".npy")
        prob_evolutions.append([prob, name])
    sum = np.array(
        np.zeros(shape=prob_evolutions[0][0].shape, dtype=prob_evolutions[0][0].dtype).tolist()
    )
    for evolution in prob_evolutions:
        sum = np.add(sum, np.array(evolution[0].tolist()))
    prob_evolutions.append([sum, "Sum"])

    plot.plot_probability_evolution(
        out_dir=None,
        probability_evolutions=prob_evolutions,
        file_name=None,
        delta_t=delta_t,
        title="Volume probability evolution",
        y_label="Probability",
        show_fig=True,
        y_min=-0.1,
        y_max=1.1,
    )

    print("\nProbability currents:\n")
    prob_current_evolutions = []
    for f in glob.glob(os.path.join(out_folder, "probability_current_evolution_*.npy")):
        print(f)
        prob_current = np.load(f)
        print(f"Sample count: {prob_current.size}")
        name = f.strip("\\probability_current_evolution_").strip(".npy")
        prob_current_evolutions.append([prob_current, name])
    plot.plot_probability_evolution(
        out_dir=None,
        probability_evolutions=prob_current_evolutions,
        file_name=None,
        delta_t=delta_t,
        title="Probability current evolution",
        y_label="Probability current",
        show_fig=True,
        y_min=-1.0,
        y_max=1.0,
    )

    print("\nIntegrated probability currents:\n")
    int_prob_current_evolutions = []
    for f in glob.glob(os.path.join(out_folder, "integrated_probability_current_*.npy")):
        print(f)
        int_prob_current = np.load(f)
        print(f"Sample count: {int_prob_current.size}")
        name = f.strip("\\integrated_probability_current_").strip(".npy")
        int_prob_current_evolutions.append([int_prob_current, name])
    plot.plot_probability_evolution(
        out_dir=None,
        probability_evolutions=int_prob_current_evolutions,
        file_name=None,
        title="Integrated probability current evolution",
        y_label="Probability",
        delta_t=delta_t,
        show_fig=True,
        y_min=-1.1,
        y_max=1.1,
    )


if __name__ == "__main__":
    main()