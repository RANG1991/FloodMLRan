from matplotlib import pyplot as plt
import re
import numpy as np


def create_dict_basin_id_to_NSE(logs_filename):
    dicts_models_dict = {}
    model_name = "empty_model_name"
    with open(logs_filename, "r", encoding="utf-8") as f:
        for row in f:
            match_model_name_string = re.search("running with model: (.*?)\n", row)
            if match_model_name_string:
                new_model_name = match_model_name_string.group(1)
                if new_model_name != model_name:
                    print(f"moving to new model with name: {new_model_name}")
                    dicts_models_dict[new_model_name] = {}
                    model_name = new_model_name
            match_nse_string = re.search("station with id: (.*?) has nse of: (.*?)\n", row)
            if match_nse_string:
                basin_id = match_nse_string.group(1)
                basin_nse = float(match_nse_string.group(2))
                dicts_models_dict[model_name][basin_id] = basin_nse
    return dicts_models_dict


def plot_CDF_NSE_basins(dict_basins_mean_NSE_loss, model_name):
    nse_losses = []
    for basin_id, mean_nes_loss in dict_basins_mean_NSE_loss.items():
        nse_losses.append(mean_nes_loss)
    nse_losses_np = np.array(nse_losses)
    # taken from https://stackoverflow.com/questions/15408371/cumulative-distribution-plots-python
    # evaluate the histogram
    values, base = np.histogram(nse_losses_np, bins=100000)
    # evaluate the cumulative
    cumulative = np.cumsum(values)
    cumulative = (cumulative - np.min(cumulative)) / np.max(cumulative)
    plt.xscale("symlog")
    plt.plot(base[:-1], cumulative, label=model_name)


def main():
    input_file_name = "slurm-5786593.out"
    plot_title = f"NSE CDF of process ID - " \
                 f"{input_file_name.replace('slurm-', '').replace('.out', '')}"
    d = create_dict_basin_id_to_NSE(input_file_name)
    for model_name in d.keys():
        dict_basins_id_to_mean_nse_loss = {}
        for basin_id, basin_nse in d[model_name].items():
            dict_basins_id_to_mean_nse_loss[basin_id] = basin_nse
        plot_CDF_NSE_basins(dict_basins_id_to_mean_nse_loss, model_name=model_name)
    if plot_title != "":
        plt.title(plot_title)
    plt.legend()
    plt.grid()
    plt.savefig(plot_title)
    plt.clf()


if __name__ == "__main__":
    main()
