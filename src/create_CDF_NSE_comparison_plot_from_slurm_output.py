import pandas as pd
import torch
from matplotlib import pyplot as plt
import re
import numpy as np
from scipy.special import softmax
from pathlib import Path
import json
import statistics
from matplotlib.pyplot import figure
import seaborn as sns
import itertools

KEYS_FROM_PARAMS_DICT = ["batch_size",
                         "num_epochs",
                         "num_hidden_units",
                         "sequence_length",
                         "sequence_length_spatial",
                         "use_all_static_attr",
                         "use_only_precip_feature",
                         "use_random_noise_spatial",
                         "use_zeros_spatial"]

COLORS_LIST = ["blue", "red", "black", "orange"]

LIST_ALLOWED_NUMBER_OF_BASINS = [135]


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


def calc_best_nse_per_model_and_num_basins(models_basins_nse_dict):
    run_numbers = list(set([run_number for _, run_number, _, _ in models_basins_nse_dict.keys()]))
    model_names = list(set([model_name for model_name, _, _, _ in models_basins_nse_dict.keys()]))
    epoch_numbers = list(set([epoch_num for _, _, epoch_num, _ in models_basins_nse_dict.keys()]))
    params_dicts = list(set([params_dict for _, _, _, params_dict in models_basins_nse_dict.keys()]))
    dict_max_median_nse_list_for_each_run = {}
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
                    if len(basins_tuple) not in LIST_ALLOWED_NUMBER_OF_BASINS:
                        continue
                    if (model_name, run_number, basins_tuple,
                        params_dict) not in dict_max_median_nse_list_for_each_run.keys():
                        dict_max_median_nse_list_for_each_run[
                            (model_name, run_number, basins_tuple, params_dict)] = list_nse
                    else:
                        if median_nse > statistics.median(
                                dict_max_median_nse_list_for_each_run[
                                    (model_name, run_number, basins_tuple, params_dict)]):
                            dict_max_median_nse_list_for_each_run[
                                (model_name, run_number, basins_tuple, params_dict)] = list_nse
                            print(f"best epoch until now is: {epoch_num}")
    dict_of_lists_max_median_nse_list_for_each_run = {}
    for model_name_in_list in model_names:
        for params_dict_in_list in params_dicts:
            dict_of_lists_max_median_nse_list_for_each_run[(model_name_in_list, params_dict_in_list)] = []
            for model_name_in_dict, run_number, basins_tuple, params_dict_in_dict in dict_max_median_nse_list_for_each_run.keys():
                if model_name_in_dict == model_name_in_list and params_dict_in_dict == params_dict_in_list and len(
                        basins_tuple) in LIST_ALLOWED_NUMBER_OF_BASINS:
                    dict_of_lists_max_median_nse_list_for_each_run[(model_name_in_list, params_dict_in_list)].append(
                        dict_max_median_nse_list_for_each_run[
                            (model_name_in_list, run_number, basins_tuple, params_dict_in_dict)])
    return dict_max_median_nse_list_for_each_run, dict_of_lists_max_median_nse_list_for_each_run


def create_dict_basin_id_to_NSE_my_code(logs_filename):
    slurm_process_id = logs_filename.name.replace('slurm-', '').replace('.out', '')
    models_basins_nse_dict = {}
    model_name = "empty_model_name"
    run_number = 0
    epoch_num = 1
    params_tuple = ()
    dict_params = {}
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
                    models_basins_nse_dict[
                        (model_name, f"{new_run_number}_{slurm_process_id}", epoch_num, params_tuple)] = {}
                    print(f"run number: {run_number}")
            # match_model_name_string = re.search("running with model: (.*?)\n", row)
            new_model_name = dict_params.get("model_name", None)
            if new_model_name:
                # new_model_name = match_model_name_string.group(1)
                if dict_params["use_only_precip_feature"]:
                    new_model_name = f"{new_model_name}\nwith daily precipitation input only"
                if dict_params["use_random_noise_spatial"]:
                    new_model_name = f"{new_model_name}\nwith random noise as spatial input"
                if dict_params["use_zeros_spatial"]:
                    new_model_name = f"{new_model_name}\nwith all zeros as spatial input"
                if new_model_name != model_name:
                    model_name = new_model_name
                    epoch_num = 1
                    models_basins_nse_dict[
                        (model_name, f"{run_number}_{slurm_process_id}", epoch_num, params_tuple)] = {}
            match_nse_string = re.search("station with id: (.*?) has nse of: (.*?)\n", row)
            if match_nse_string:
                basin_id = match_nse_string.group(1)
                basin_nse = float(match_nse_string.group(2))
                models_basins_nse_dict[(model_name, f"{run_number}_{slurm_process_id}", epoch_num, params_tuple)][
                    basin_id] = basin_nse
            match_best_nse_so_far_string = re.search("best median NSE so far: (.*?)\n", row)
            if match_best_nse_so_far_string:
                epoch_num += 1
                models_basins_nse_dict[(model_name, f"{run_number}_{slurm_process_id}", epoch_num, params_tuple)] = {}
    return models_basins_nse_dict


def calc_dicts_from_all_runs_and_all_files(input_file_paths):
    dict_all_runs_from_all_files = {}
    dict_avg_runs_from_all_files = {}
    for input_file_path in input_file_paths:
        d = create_dict_basin_id_to_NSE_my_code(input_file_path)
        dict_all_runs_single_file, dict_nse_list_per_run_single_file = calc_best_nse_per_model_and_num_basins(d)
        dict_all_runs_from_all_files.update(dict_all_runs_single_file)
        for (model_name_in_list, params_dict_in_list) in dict_nse_list_per_run_single_file.keys():
            if (model_name_in_list, params_dict_in_list) not in dict_avg_runs_from_all_files.keys():
                dict_avg_runs_from_all_files[(model_name_in_list, params_dict_in_list)] = []
            dict_avg_runs_from_all_files[(model_name_in_list, params_dict_in_list)].append(
                dict_nse_list_per_run_single_file[(model_name_in_list, params_dict_in_list)])
    for (model_name_in_list, params_dict_in_list) in dict_avg_runs_from_all_files.keys():
        list_of_nse_lists_for_one_model = dict_avg_runs_from_all_files[(model_name_in_list, params_dict_in_list)]
        dict_avg_runs_from_all_files[(model_name_in_list, params_dict_in_list)] = np.mean(
            np.vstack(list_of_nse_lists_for_one_model), axis=0)
    return dict_all_runs_from_all_files, dict_avg_runs_from_all_files


def plot_NSE_CDF_graphs_all_runs(run_numbers, all_basins_tuples, params_dicts, model_name,
                                 dict_all_runs_from_all_files, dict_avg_runs_from_all_files):
    for run_number in run_numbers:
        for basin_tuple in all_basins_tuples:
            for params_dict in params_dicts:
                if (model_name, run_number, basin_tuple, params_dict) not in dict_all_runs_from_all_files.keys():
                    continue
                print(f"number of basins of model with name: {model_name} "
                      f"and run number: {run_number} is: {len(basin_tuple)}")
                if len(basin_tuple) != 135:
                    continue
                figure(figsize=(14, 12))
                plt.grid()
                plot_CDF_NSE_basins(dict_all_runs_from_all_files[(model_name, run_number, basin_tuple, params_dict)],
                                    params_dict,
                                    model_name=model_name,
                                    plot_color=COLORS_LIST[0],
                                    plot_opacity=0.6,
                                    line_width=0.5)
                plot_CDF_NSE_basins(dict_avg_runs_from_all_files[(model_name, params_dict)],
                                    params_dict,
                                    model_name=model_name,
                                    plot_color=COLORS_LIST[0],
                                    plot_opacity=1,
                                    line_width=3,
                                    label=f"mean CDF NSE of model: {model_name.replace('_', '-')}",
                                    ablation_study=False)
                plt.savefig("NSE_CDF" + f"_{model_name}".replace('\n', ' '))
                plt.close()


def plot_NSE_CDF_graphs_average(params_dicts, model_names, dict_avg_runs_from_all_files, ablation_study=False):
    figure(figsize=(14, 12))
    for ind, model_name in enumerate(model_names):
        for params_dict in params_dicts:
            if (model_name, params_dict) not in dict_avg_runs_from_all_files.keys():
                continue
            plot_CDF_NSE_basins(dict_avg_runs_from_all_files[(model_name, params_dict)],
                                params_dict,
                                model_name=model_name,
                                plot_color=COLORS_LIST[ind],
                                plot_opacity=1,
                                line_width=3,
                                label=f"mean CDF NSE of model: {model_name.replace('_', '-')}",
                                ablation_study=ablation_study)
    plt.legend(loc='upper left')
    plt.grid()
    if ablation_study:
        plt.savefig("NSE_CDF_ablation_study")
    else:
        plt.savefig("NSE_CDF" + f"_{'_'.join(model_names)}".replace('\n', ' '))
    plt.close()


def plot_NSE_CDF_graphs_my_code(ablation_study=False):
    if ablation_study:
        input_file_names = ["slurm-19089603.out", "slurm-19100407.out", "slurm-19185354.out", "slurm-19128144.out"]
    else:
        input_file_names = ["slurm-19195517.out", "slurm-19195949.out", "slurm-19195809.out"]
        # input_file_names = ["slurm-19178982.out", "slurm-19173334.out", "slurm-19170388.out"]
    input_file_paths = [Path(f"../slurm_output_files/slurm_files_ensemble_comparison/{file_name}").resolve() for
                        file_name in input_file_names]
    dict_all_runs_from_all_files, dict_avg_runs_from_all_files = calc_dicts_from_all_runs_and_all_files(
        input_file_paths)
    all_basins_tuples = list(set([basin_tuple for _, _, basin_tuple, _ in dict_all_runs_from_all_files.keys()]))
    model_names = sorted(list(set([model_name for model_name, _, _, _ in dict_all_runs_from_all_files.keys()])))
    run_numbers = list(set([run_number for _, run_number, _, _ in dict_all_runs_from_all_files.keys()]))
    params_dicts = list(set([params_dict for _, _, _, params_dict in dict_all_runs_from_all_files.keys()]))
    input_files_names_formatted = "_".join(
        [input_file_path.name.replace('slurm-', '').replace('.out', '') for input_file_path in input_file_paths])
    plot_title = f"NSE CDF of slurm files - {input_files_names_formatted}"
    plt.rcParams.update({'font.size': 22})
    if not ablation_study:
        for model_name in model_names:
            plot_NSE_CDF_graphs_all_runs(run_numbers=run_numbers, all_basins_tuples=all_basins_tuples,
                                         params_dicts=params_dicts, model_name=model_name,
                                         dict_all_runs_from_all_files=dict_all_runs_from_all_files,
                                         dict_avg_runs_from_all_files=dict_avg_runs_from_all_files)
    plot_NSE_CDF_graphs_average(params_dicts=params_dicts, model_names=model_names,
                                dict_avg_runs_from_all_files=dict_avg_runs_from_all_files,
                                ablation_study=ablation_study)


def plot_NSE_CDF_graph_frederik_code():
    input_file_name = "slurm-5817859.out"
    model_name = "LSTM_Frederik"
    plot_title = f"NSE CDF of process ID - " \
                 f"{input_file_name.replace('slurm-', '').replace('.out', '')} with model - {model_name}"
    d = create_dict_basin_id_to_NSE_frederik_code(input_file_name)
    plot_CDF_NSE_basins(d, {},
                        model_name,
                        plot_color=COLORS_LIST[0],
                        plot_opacity=1,
                        line_width=2)
    plt.legend()
    plt.grid()
    plt.savefig(plot_title)
    plt.clf()


def plot_CDF_NSE_basins(nse_losses_np, params_tuple, model_name, plot_color, plot_opacity, line_width,
                        label="", ablation_study=False):
    # taken from https://stackoverflow.com/questions/15408371/cumulative-distribution-plots-python
    values, base = np.histogram(nse_losses_np, bins=100000)
    cumulative = np.cumsum(values)
    cumulative = (cumulative - np.min(cumulative)) / (np.max(cumulative) - np.min(cumulative))
    # plt.xscale("symlog")
    if ablation_study:
        plt.xlim((-1, 1))
    else:
        plt.xlim((0, 1))
    plt.xlabel("NSE")
    plt.ylabel("CDF")
    # sns.kdeplot(nse_losses_np, cumulative=True)
    if label != "":
        plt.plot(base[:-1], cumulative, color=plot_color, alpha=plot_opacity, linewidth=line_width,
                 label=label)
    else:
        plt.plot(base[:-1], cumulative, color=plot_color, alpha=plot_opacity, linewidth=line_width)


if __name__ == "__main__":
    plot_NSE_CDF_graphs_my_code(ablation_study=False)
