import torch
from matplotlib import pyplot as plt
import re
import numpy as np
from scipy.special import softmax
from pathlib import Path
import json
import statistics


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


def calc_best_nse_per_model_and_num_basins(models_basins_nse_dict, calc_average_nse_per_basin=False):
    run_numbers = set([run_number for _, run_number, _ in models_basins_nse_dict.keys()])
    model_names = set([model_name for model_name, _, _ in models_basins_nse_dict.keys()])
    epoch_numbers = set([epoch_num for _, _, epoch_num in models_basins_nse_dict.keys()])
    model_name_and_basins_tuple_to_best_nse = {}
    for model_name in model_names:
        for run_number in run_numbers:
            for epoch_num in epoch_numbers:
                if (model_name, run_number, epoch_num) not in models_basins_nse_dict.keys() or \
                        len(models_basins_nse_dict[(model_name, run_number, epoch_num)].items()) == 0:
                    continue
                basin_id_to_nse_list_tuples = sorted(
                    list(models_basins_nse_dict[(model_name, run_number, epoch_num)].items()),
                    key=lambda x: x[0])
                list_basin_ids, list_nse = zip(*basin_id_to_nse_list_tuples)
                list_nse = np.array(list_nse)
                median_nse = statistics.median(list_nse)
                basins_tuple = tuple(list_basin_ids)
                if (model_name, run_number, basins_tuple) not in model_name_and_basins_tuple_to_best_nse.keys():
                    if calc_average_nse_per_basin:
                        model_name_and_basins_tuple_to_best_nse[(model_name, run_number, basins_tuple)] = [list_nse]
                    else:
                        model_name_and_basins_tuple_to_best_nse[(model_name, run_number, basins_tuple)] = list_nse
                else:
                    if calc_average_nse_per_basin:
                        model_name_and_basins_tuple_to_best_nse[(model_name, run_number, basins_tuple)].append(list_nse)
                    else:
                        if median_nse > statistics.median(
                                model_name_and_basins_tuple_to_best_nse[(model_name, run_number, basins_tuple)]):
                            model_name_and_basins_tuple_to_best_nse[(model_name, run_number, basins_tuple)] = list_nse
                            print(f"best epoch until now is: {epoch_num}")
    if calc_average_nse_per_basin:
        for model_name, run_number, basins_tuple in model_name_and_basins_tuple_to_best_nse.keys():
            lists_of_nse_of_num_basins = model_name_and_basins_tuple_to_best_nse[(model_name, run_number, basins_tuple)]
            model_name_and_basins_tuple_to_best_nse[(model_name, run_number, basins_tuple)] = np.mean(
                lists_of_nse_of_num_basins, axis=1)
    return model_name_and_basins_tuple_to_best_nse


def create_dict_basin_id_to_NSE_my_code(logs_filename):
    models_basins_nse_dict = {}
    model_name = "empty_model_name"
    run_number = 0
    epoch_num = 1
    with open(logs_filename, "r", encoding="utf-8") as f:
        for row in f:
            match_run_number_string = re.search("(run number: |wandb: Agent Starting Run: )", row)
            if match_run_number_string:
                new_run_number = run_number + 1
                if new_run_number != run_number:
                    run_number = new_run_number
                    epoch_num = 1
                    models_basins_nse_dict[(model_name, new_run_number, epoch_num)] = {}
                    print(f"run number: {run_number}")
            match_model_name_string = re.search("running with model: (.*?)\n", row)
            if match_model_name_string:
                new_model_name = match_model_name_string.group(1)
                if new_model_name != model_name:
                    model_name = new_model_name
                    epoch_num = 1
                    models_basins_nse_dict[(new_model_name, run_number, epoch_num)] = {}
            match_nse_string = re.search("station with id: (.*?) has nse of: (.*?)\n", row)
            if match_nse_string:
                basin_id = match_nse_string.group(1)
                basin_nse = float(match_nse_string.group(2))
                models_basins_nse_dict[(model_name, run_number, epoch_num)][basin_id] = basin_nse
            match_best_nse_so_far_string = re.search("best median NSE so far: (.*?)\n", row)
            if match_best_nse_so_far_string:
                epoch_num += 1
                models_basins_nse_dict[(new_model_name, run_number, epoch_num)] = {}
    return models_basins_nse_dict


def plot_NSE_CDF_graphs_my_code():
    input_file_names = ["../slurm-16186727.out"]
    input_file_paths = [Path(file_name).resolve() for file_name in input_file_names]
    dict_all_files = {}
    for input_file_path in input_file_paths:
        d = create_dict_basin_id_to_NSE_my_code(f"{input_file_path}")
        d = calc_best_nse_per_model_and_num_basins(d)
        dict_all_files.update(d)
    all_basins_tuples = set([basin_tuple for _, _, basin_tuple in dict_all_files.keys()])
    model_names = set([model_name for model_name, _, _ in dict_all_files.keys()])
    run_numbers = set([run_number for _, run_number, _ in dict_all_files.keys()])
    input_files_names_formatted = "_".join(
        [input_file_path.name.replace('slurm-', '').replace('.out', '') for input_file_path in input_file_paths])
    plot_title = f"Comparison of different configurations of CNN_LSTM and LSTM"
    for model_name in model_names:
        for run_number in run_numbers:
            for basin_tuple in all_basins_tuples:
                if (model_name, run_number, basin_tuple) not in dict_all_files.keys():
                    continue
                print(f"number of basins of model with name: {model_name} "
                      f"and run number: {run_number} is: {len(basin_tuple)}")
                plot_CDF_NSE_basins(dict_all_files[(model_name, run_number, basin_tuple)], model_name=model_name,
                                    num_basins=len(basin_tuple))
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


def plot_CDF_NSE_basins(nse_losses_np, model_name, num_basins):
    # taken from https://stackoverflow.com/questions/15408371/cumulative-distribution-plots-python
    # evaluate the histogram
    values, base = np.histogram(nse_losses_np, bins=100000)
    # evaluate the cumulative
    cumulative = np.cumsum(values)
    cumulative = (cumulative - np.min(cumulative)) / np.max(cumulative)
    # plt.xscale("symlog")
    plt.xlim((0, 1))
    plt.xlabel("NSE")
    plt.ylabel("CDF")
    plt.plot(base[:-1], cumulative, label=f"model name: {model_name} number of basins: {num_basins}")


def main():
    plot_NSE_CDF_graphs_my_code()


if __name__ == "__main__":
    # d = {"dataset": "CAMELS", "optim": "Adam", "num_epochs": 10,
    # "sequence_length_spatial": 14, "num_processes_ddp": 3,
    #      "limit_size_above_1000": "True", "num_workers_data_loader": 2, "batch_size": 1024}
    # with open("config_files_yml/config_run_above_1000_basins.json", "w") as f:
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
