from matplotlib import pyplot as plt
import re
import numpy as np
import glob
import pathlib

PATTERN = r"The NSE loss for basin with ID: (\d+) is: ([-+]?\d*[.,]?\d*)"


def create_dict_basin_id_to_NSE(logs_filename):
    d = {}
    with open(logs_filename, "r") as f:
        for row in f:
            basin_nse_loss_str = re.search(PATTERN, row)
            if basin_nse_loss_str is not None:
                basin_id = basin_nse_loss_str.group(1)
                nse_loss = basin_nse_loss_str.group(2)
                if basin_id not in d.keys():
                    d[basin_id] = []
                try:
                    d[basin_id].append(float(nse_loss))
                except ValueError as e:
                    pass
    return d


def plot_CDF_NSE_basins(dict_basins_mean_NSE_loss, graph_color):
    nse_losses = []
    for basin_id, mean_nes_loss in dict_basins_mean_NSE_loss.items():
        if mean_nes_loss >= 0:
            nse_losses.append(mean_nes_loss)
    nse_losses_np = np.array(nse_losses)
    # taken from https://stackoverflow.com/questions/15408371/cumulative-distribution-plots-python
    # evaluate the histogram
    values, base = np.histogram(nse_losses_np, bins=100)
    # evaluate the cumulative
    cumulative = np.cumsum(values)
    cumulative = (cumulative - np.min(cumulative)) / np.max(cumulative)
    # plot the cumulative function
    plt.plot(base[:-1], cumulative, c=graph_color)


def parse_output_file(output_file_path, dataset_name, graph_color):
    with open(f"nse_per_basin_{dataset_name}.txt", "w") as f:
        d = create_dict_basin_id_to_NSE(output_file_path)
        dict_basins_id_to_mean_nse_loss = {}
        for basin_id, nse_losses_list in d.items():
            print("Basin ID: {}, NSE losses: {}, mean NSE losses: {}".format(basin_id, nse_losses_list,
                                                                             sum(nse_losses_list) / len(
                                                                                 nse_losses_list)))
            f.write("Basin ID: {}, NSE losses: {}, mean NSE losses: {}\n".format(basin_id, nse_losses_list,
                                                                                 sum(nse_losses_list) / len(
                                                                                     nse_losses_list)))
            dict_basins_id_to_mean_nse_loss[basin_id] = sum(nse_losses_list) / len(nse_losses_list)
    plot_CDF_NSE_basins(dict_basins_id_to_mean_nse_loss, graph_color)


def main():
    output_files = glob.glob("output_*.log")
    dataset_names = []
    colors = ["red", "blue", "yellow"]
    for i in range(len(output_files)):
        dataset_name = str(pathlib.Path(output_files[i]).stem).replace("output_", "")
        dataset_names.append(dataset_name)
        parse_output_file(output_file_path=output_files[i], dataset_name=dataset_name,
                          graph_color=colors[i % len(output_files)])
    plt.legend(dataset_names)
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
