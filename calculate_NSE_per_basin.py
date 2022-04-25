from matplotlib import pyplot as plt
import re
import numpy as np

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
                d[basin_id].append(float(nse_loss))
    return d


def plot_CDF_NSE_basins(dict_basins_mean_NSE_loss):
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
    plt.plot(base[:-1], cumulative, c='blue')
    plt.grid()
    plt.show()


def main():
    with open("./nse_per_basin.txt", "w") as f:
        d = create_dict_basin_id_to_NSE("./output.log")
        dict_basins_id_to_mean_nse_loss = {}
        for basin_id, nse_losses_list in d.items():
            print("Basin ID: {}, NSE losses: {}, mean NSE losses: {}".format(basin_id, nse_losses_list,
                                                                             sum(nse_losses_list) / len(
                                                                                 nse_losses_list)))
            f.write("Basin ID: {}, NSE losses: {}, mean NSE losses: {}\n".format(basin_id, nse_losses_list,
                                                                                 sum(nse_losses_list) / len(
                                                                                     nse_losses_list)))
            dict_basins_id_to_mean_nse_loss[basin_id] = sum(nse_losses_list) / len(nse_losses_list)
    plot_CDF_NSE_basins(dict_basins_id_to_mean_nse_loss)


if __name__ == "__main__":
    main()
