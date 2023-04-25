import torch
from matplotlib import pyplot as plt
import re
import numpy as np
from scipy.special import softmax
from pathlib import Path
import json


def create_dict_basin_id_to_NSE_frederik_code(logs_filename):
    dicts_models_dict = {}
    with open(logs_filename, "r", encoding="utf-8") as f:
        for row in f:
            match_nse_string = re.search("The NSE loss for basin with ID: (.*?) is: (.*?)\n", row)
            if match_nse_string:
                basin_id = match_nse_string.group(1)
                basin_nse = float(match_nse_string.group(2))
                dicts_models_dict[basin_id] = basin_nse
    return dicts_models_dict


def create_dict_basin_id_to_NSE_my_code(logs_filename):
    dicts_models_dict = {}
    model_name = "empty_model_name"
    run_number = 1
    with open(logs_filename, "r", encoding="utf-8") as f:
        for row in f:
            match_run_number_string = re.search("run number: (.*?)\n", row)
            if match_run_number_string:
                new_run_number = match_run_number_string.group(1)
                if new_run_number != run_number:
                    print(f"moving to new run number: {new_run_number}")
                    dicts_models_dict[(model_name, new_run_number)] = {}
                    run_number = new_run_number
            match_model_name_string = re.search("running with model: (.*?)\n", row)
            if match_model_name_string:
                new_model_name = match_model_name_string.group(1)
                if new_model_name != model_name:
                    print(f"moving to new model with name: {new_model_name}")
                    dicts_models_dict[(new_model_name, run_number)] = {}
                    model_name = new_model_name
            match_nse_string = re.search("station with id: (.*?) has nse of: (.*?)\n", row)
            if match_nse_string:
                basin_id = match_nse_string.group(1)
                basin_nse = float(match_nse_string.group(2))
                dicts_models_dict[(model_name, run_number)][basin_id] = basin_nse
    return dicts_models_dict


def plot_NSE_CDF_graphs_my_code():
    input_file_name = Path("../slurm-6522830.out").resolve()
    d = create_dict_basin_id_to_NSE_my_code(f"{input_file_name}")
    run_numbers = set([run_number for _, run_number in d.keys()])
    model_names = set([model_name for model_name, _ in d.keys()])
    plot_title = f"NSE CDF of process ID - " \
                 f"{input_file_name.name.replace('slurm-', '').replace('.out', '')}"
    for model_name in model_names:
        # plot_title = f"NSE CDF of process ID - " \
        #              f"{input_file_name.replace('slurm-', '').replace('.out', '')} with model - {model_name}"
        for run_number in run_numbers:
            if (model_name, run_number) not in d.keys() or len(d[(model_name, run_number)].items()) == 0:
                continue
            dict_basins_id_to_mean_nse_loss = {}
            for basin_id, basin_nse in d[(model_name, run_number)].items():
                dict_basins_id_to_mean_nse_loss[basin_id] = basin_nse
            plot_CDF_NSE_basins(dict_basins_id_to_mean_nse_loss, model_name=model_name, run_number=run_number)
    if plot_title != "":
        plt.title(plot_title)
    plt.legend()
    plt.grid()
    plt.savefig(plot_title)
    plt.clf()


def plot_NSE_CDF_graph_frederik_code():
    input_file_name = "slurm-5817859.out"
    model_name = "LSTM_Frederik"
    plot_title = f"NSE CDF of process ID - " \
                 f"{input_file_name.replace('slurm-', '').replace('.out', '')} with model - {model_name}"
    d = create_dict_basin_id_to_NSE_frederik_code(input_file_name)
    plot_CDF_NSE_basins(d, model_name, 1)
    plt.legend()
    plt.grid()
    plt.savefig(plot_title)
    plt.clf()


def plot_CDF_NSE_basins(dict_basins_mean_NSE_loss, model_name, run_number):
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
    # plt.xscale("symlog")
    plt.xlim((0, 1))
    plt.plot(base[:-1], cumulative, label=f"model name: {model_name} run number: {run_number}")


def main():
    plot_NSE_CDF_graphs_my_code()


if __name__ == "__main__":
    # d = {"dataset": "CAMELS", "optim": "Adam", "num_epochs": 10, "sequence_length_spatial": 14, "num_processes_ddp": 3,
    #      "limit_size_above_1000": "True", "num_workers_data_loader": 2, "batch_size": 1024}
    # with open("config_files_json/config_run_above_1000_basins.json", "w") as f:
    #     json.dump(d, f, indent=4)
    # query = np.array([[0.5, 0.1, 0.2], [0.7, 0.8, 0.2]])
    # key = np.array([[1.0, 0.2, 0.4], [1.4, 1.6, 0.4]])
    # value = np.array([[1.5, 0.3, 0.6], [2.1, 2.4, 0.6]])
    # softmax_out = softmax(np.matmul(query, key.T) / np.sqrt(3), axis=1)
    # res = np.matmul(softmax_out, value)
    # for i in range(softmax_out.shape[0]):
    #     for j in range(softmax_out.shape[1]):
    #         print("{0:0.2f}".format(softmax_out[i, j]), end=" ")
    #     print()
    # for i in range(res.shape[0]):
    #     for j in range(res.shape[1]):
    #         print("{0:0.2f}".format(res[i, j]), end=" ")
    #     print()
    main()
