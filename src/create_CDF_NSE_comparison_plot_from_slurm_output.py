import torch
from matplotlib import pyplot as plt
import re
import numpy as np
from scipy.special import softmax
from pathlib import Path
import json
import statistics
from matplotlib.pyplot import figure

KEYS_FROM_PARAMS_DICT = ["batch_size",
                         "num_epochs",
                         "num_hidden_units",
                         "sequence_length",
                         "sequence_length_spatial",
                         "use_all_static_attr",
                         "use_only_precip_feature",
                         "use_random_noise_spatial",
                         "use_zeros_spatial"]


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


def calc_best_nse_per_model_and_num_basins(models_basins_nse_dict, calc_average_runs_in_model=False):
    run_numbers = set([run_number for _, run_number, _, _ in models_basins_nse_dict.keys()])
    model_names = set([model_name for model_name, _, _, _ in models_basins_nse_dict.keys()])
    epoch_numbers = set([epoch_num for _, _, epoch_num, _ in models_basins_nse_dict.keys()])
    params_dicts = set([params_dict for _, _, _, params_dict in models_basins_nse_dict.keys()])
    model_name_and_basins_tuple_to_best_nse = {}
    for model_name in model_names:
        for run_number in run_numbers:
            for epoch_num in epoch_numbers:
                for params_dict in params_dicts:
                    if (model_name, run_number, epoch_num, params_dict) not in models_basins_nse_dict.keys() or \
                            len(models_basins_nse_dict[(model_name, run_number, epoch_num, params_dict)].items()) == 0:
                        continue
                    basin_id_to_nse_list_tuples = sorted(
                        list(models_basins_nse_dict[(model_name, run_number, epoch_num, params_dict)].items()),
                        key=lambda x: x[0])
                    list_basin_ids, list_nse = zip(*basin_id_to_nse_list_tuples)
                    list_nse = np.array(list_nse)
                    median_nse = statistics.median(list_nse)
                    basins_tuple = tuple(list_basin_ids)
                    if len(basins_tuple) != 135:
                        continue
                    if (model_name, run_number, basins_tuple,
                        params_dict) not in model_name_and_basins_tuple_to_best_nse.keys():
                        model_name_and_basins_tuple_to_best_nse[
                            (model_name, run_number, basins_tuple, params_dict)] = list_nse
                    else:
                        if median_nse > statistics.median(
                                model_name_and_basins_tuple_to_best_nse[
                                    (model_name, run_number, basins_tuple, params_dict)]):
                            model_name_and_basins_tuple_to_best_nse[
                                (model_name, run_number, basins_tuple, params_dict)] = list_nse
                            print(f"best epoch until now is: {epoch_num}")
    if calc_average_runs_in_model:
        model_name_and_params_dict_to_nse_lists = {}
        model_name_and_params_dict_to_avg_nse = {}
        for model_name_in_list in model_names:
            for params_dict_in_list in params_dicts:
                model_name_and_params_dict_to_nse_lists[(model_name_in_list, params_dict_in_list)] = []
                for model_name_in_dict, run_number, basins_tuple, params_dict_in_dict in model_name_and_basins_tuple_to_best_nse.keys():
                    if model_name_in_dict == model_name_in_list and params_dict_in_dict == params_dict_in_list and len(
                            basins_tuple) == 135:
                        model_name_and_params_dict_to_nse_lists[(model_name_in_list, params_dict_in_list)].append(
                            model_name_and_basins_tuple_to_best_nse[
                                (model_name_in_list, run_number, basins_tuple, params_dict_in_dict)])
                nse_lists = model_name_and_params_dict_to_nse_lists[(model_name_in_list, params_dict_in_list)]
                if len(nse_lists) > 0:
                    nse_lists_concat = np.vstack(nse_lists)
                    model_name_and_params_dict_to_avg_nse[(model_name_in_list, params_dict_in_list)] = np.max(
                        nse_lists_concat, axis=0)
        return model_name_and_params_dict_to_avg_nse
    else:
        return model_name_and_basins_tuple_to_best_nse


def create_dict_basin_id_to_NSE_my_code(logs_filename):
    models_basins_nse_dict = {}
    model_name = "empty_model_name"
    run_number = 0
    epoch_num = 1
    params_tuple = ()
    with open(logs_filename, "r", encoding="utf-8") as f:
        for row in f:
            match_parameters_dict = re.search(r"running with parameters: \{", row)
            if match_parameters_dict:
                row = next(f)
                params_tuple_as_str = "{"
                while row != "}\n":
                    params_tuple_as_str += row.replace("\n", "")
                    row = next(f)
                params_tuple_as_str += "}"
                dict_params = json.loads(params_tuple_as_str)
                params_tuple = tuple((k, dict_params.get(k, None)) for k in KEYS_FROM_PARAMS_DICT)
            match_run_number_string = re.search("(run number: |wandb: Agent Starting Run: )", row)
            if match_run_number_string:
                new_run_number = run_number + 1
                if new_run_number != run_number:
                    run_number = new_run_number
                    epoch_num = 1
                    models_basins_nse_dict[(model_name, new_run_number, epoch_num, params_tuple)] = {}
                    print(f"run number: {run_number}")
            match_model_name_string = re.search("running with model: (.*?)\n", row)
            if match_model_name_string:
                new_model_name = match_model_name_string.group(1)
                if new_model_name != model_name:
                    model_name = new_model_name
                epoch_num = 1
                models_basins_nse_dict[(model_name, run_number, epoch_num, params_tuple)] = {}
            match_nse_string = re.search("station with id: (.*?) has nse of: (.*?)\n", row)
            if match_nse_string:
                basin_id = match_nse_string.group(1)
                basin_nse = float(match_nse_string.group(2))
                models_basins_nse_dict[(model_name, run_number, epoch_num, params_tuple)][basin_id] = basin_nse
            match_best_nse_so_far_string = re.search("best median NSE so far: (.*?)\n", row)
            if match_best_nse_so_far_string:
                epoch_num += 1
                models_basins_nse_dict[(model_name, run_number, epoch_num, params_tuple)] = {}
    return models_basins_nse_dict


def plot_NSE_CDF_graphs_my_code():
    calc_average_runs_in_model = False
    input_file_names = ["slurm-17476442.out", "slurm-17477923.out"]
    input_file_paths = [Path("../slurm_output_files/" + file_name).resolve() for file_name in input_file_names]
    dict_all_files = {}
    for input_file_path in input_file_paths:
        d = create_dict_basin_id_to_NSE_my_code(f"{input_file_path}")
        d = calc_best_nse_per_model_and_num_basins(d, calc_average_runs_in_model=calc_average_runs_in_model)
        dict_all_files.update(d)
    if not calc_average_runs_in_model:
        all_basins_tuples = set([basin_tuple for _, _, basin_tuple, _ in dict_all_files.keys()])
        model_names = set([model_name for model_name, _, _, _ in dict_all_files.keys()])
        run_numbers = set([run_number for _, run_number, _, _ in dict_all_files.keys()])
        params_dicts = set([params_dict for _, _, _, params_dict in dict_all_files.keys()])
        input_files_names_formatted = "_".join(
            [input_file_path.name.replace('slurm-', '').replace('.out', '') for input_file_path in input_file_paths])
        plot_title = f"NSE CDF of slurm files - {input_files_names_formatted}"
        figure(figsize=(20, 16))
        for model_name in model_names:
            for run_number in run_numbers:
                for basin_tuple in all_basins_tuples:
                    for params_dict in params_dicts:
                        if (model_name, run_number, basin_tuple, params_dict) not in dict_all_files.keys():
                            continue
                        print(f"number of basins of model with name: {model_name} "
                              f"and run number: {run_number} is: {len(basin_tuple)}")
                        if len(basin_tuple) != 135:
                            continue
                        plot_CDF_NSE_basins(dict_all_files[(model_name, run_number, basin_tuple, params_dict)],
                                            params_dict,
                                            model_name=model_name,
                                            num_basins=len(basin_tuple))
    else:
        model_names = set([model_name for model_name, _ in dict_all_files.keys()])
        params_dicts = set([params_dict for _, params_dict in dict_all_files.keys()])
        input_files_names_formatted = "_".join(
            [input_file_path.name.replace('slurm-', '').replace('.out', '') for input_file_path in input_file_paths])
        plot_title = f"NSE CDF of slurm files - {input_files_names_formatted}"
        figure(figsize=(16, 12))
        for model_name in model_names:
            for params_dict in params_dicts:
                if (model_name, params_dict) not in dict_all_files.keys():
                    continue
                print(f"number of basins of model with name: {model_name}")
                plot_CDF_NSE_basins(dict_all_files[(model_name, params_dict)],
                                    num_basins=135,
                                    model_name=model_name,
                                    params_tuple=params_dict)
    if plot_title != "":
        plt.title(plot_title)
    plt.legend(loc='upper left')
    plt.grid()
    plt.savefig(plot_title)
    plt.clf()


def plot_NSE_CDF_graph_frederik_code():
    input_file_name = "slurm-5817859.out"
    model_name = "LSTM_Frederik"
    plot_title = f"NSE CDF of process ID - " \
                 f"{input_file_name.replace('slurm-', '').replace('.out', '')} with model - {model_name}"
    d = create_dict_basin_id_to_NSE_frederik_code(input_file_name)
    plot_CDF_NSE_basins(d, {}, model_name, 1)
    plt.legend()
    plt.grid()
    plt.savefig(plot_title)
    plt.clf()


def plot_CDF_NSE_basins(nse_losses_np, params_tuple, model_name, num_basins):
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
    plt.plot(base[:-1], cumulative,
             label=f"model name: {model_name}; number of basins: {num_basins}; "
                   f"params_tuple:{json.dumps(dict(params_tuple), indent=4)}")


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
