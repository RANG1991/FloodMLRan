import os
import sys
from torch.utils.data import DataLoader
import ERA5_dataset
import CAMELS_dataset
import FloodML_transformer
from tqdm import tqdm
import FloodML_lstm
import torch.optim
import torch.nn as nn
from os import listdir
from os.path import isfile, join
import pandas as pd
import matplotlib
import math
import matplotlib.pyplot as plt
from typing import Tuple
import numpy as np
import itertools
from datetime import datetime
import statistics
from random import shuffle
import argparse

matplotlib.use("AGG")

K_VALUE_CROSS_VALIDATION = 2


def train_epoch(model, optimizer, loader, loss_func, epoch, device):
    # set model to train mode (important for dropout)
    model.train()
    pbar = tqdm(loader, file=sys.stdout)
    print(f"Epoch {epoch}")
    pbar.set_description(f"Epoch {epoch}")
    # request mini-batch of data from the loader
    running_loss = 0.0
    for _, xs, ys in pbar:
        # delete previously stored gradients from the model
        optimizer.zero_grad()
        # push data to GPU (if available)
        xs, ys = xs.to(device), ys.to(device)
        # get model predictions
        y_hat = model(xs)
        # calculate loss
        loss = loss_func(y_hat.squeeze(0), ys)
        # calculate gradients
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        # update the weights
        optimizer.step()
        # write current loss in the progress bar
        running_loss += loss.item()
        # print(f"Loss: {loss.item():.4f}")
        # pbar.set_postfix_str(f"Loss: {loss.item():.4f}")
    print(f"Loss on the entire training epoch: {(running_loss / len(loader)):.4f}")
    return running_loss / (len(loader))


def eval_model(
    model, loader, device, preds_obs_dict_per_basin, loss_func
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the model.

    :param model: A torch.nn.Module implementing the LSTM model
    :param loader: A PyTorch DataLoader, providing the data.

    :return: Two torch Tensors, containing the observations and
        model predictions

    Parameters
    ----------
    loss_func
    device
    preds_obs_dict_per_basin
    """
    # set model to eval mode (important for dropout)
    model.eval()
    # in inference mode, we don't need to store intermediate steps for backprob
    running_loss = 0.0
    with torch.no_grad():
        # request mini-batch of data from the loader
        for station_id_batch, xs, ys in loader:
            station_id = station_id_batch[0]
            if station_id_batch not in preds_obs_dict_per_basin.keys():
                preds_obs_dict_per_basin[station_id] = []
            # push data to GPU (if available)
            xs = xs.to(device)
            # get model predictions
            y_hat = model(xs)
            loss = loss_func(y_hat.squeeze(0), ys.to(device))
            running_loss += loss.item()
            preds_obs_dict_per_basin[station_id].append((ys, y_hat))
    print(
        f"Loss on the entire evaluation (test or validation) epoch: {(running_loss / len(loader)):.4f}"
    )
    return running_loss / (len(loader))


def calc_nse(obs: np.array, sim: np.array) -> float:
    """Calculate Nash-Shutcliff-Efficiency.

    :param obs: Array containing the observations
    :param sim: Array containing the simulations
    :return: NSE value.
    """
    # only consider time steps, where observations are available
    # COMMENT FROM EFRAT TO RONEN: NEGATIVE VALUES ARE FINE! I COMMENTED THE TWO LINES BELOW
    # sim = np.delete(sim, np.argwhere(obs < 0), axis=0)
    # obs = np.delete(obs, np.argwhere(obs < 0), axis=0)
    # check for NaNs in observations
    sim = np.delete(sim, np.argwhere(np.isnan(obs)), axis=0)
    obs = np.delete(obs, np.argwhere(np.isnan(obs)), axis=0)
    denominator = np.sum((obs - np.mean(obs)) ** 2)
    numerator = np.sum((sim - obs) ** 2)
    nse_val = 1 - (numerator / denominator)
    return nse_val


def calc_validation_basins_nse(
    preds_obs_dict_per_basin, num_epoch, num_basins_for_nse_calc=10
):
    stations_ids = list(preds_obs_dict_per_basin.keys())
    nse_list_basins = []
    max_nse = -1
    max_basin = -1
    max_preds = []
    max_obs = []
    for stations_id in stations_ids:
        obs_and_preds = preds_obs_dict_per_basin[stations_id]
        obs, preds = zip(*obs_and_preds)
        obs = np.array([single_obs.cpu().numpy() for single_obs in obs])
        preds = np.array([single_pred.cpu().numpy() for single_pred in preds])
        nse = calc_nse(obs, preds)
        # print(f"station with id: {stations_id} has nse of: {nse}")
        nse_list_basins.append(nse)
        if nse > max_nse or max_nse == -1:
            max_nse = nse
            max_basin = stations_id
            max_preds = preds
            max_obs = obs
    curr_datetime = datetime.now()
    curr_datetime_str = curr_datetime.strftime("%d-%m-%Y_%H:%M:%S")
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.plot(max_obs.squeeze(), label="observation")
    ax.plot(max_preds.squeeze(), label="prediction")
    ax.legend()
    ax.set_title(f"Basin {max_basin} - NSE: {max_nse:.3f}")
    plt.savefig(
        f"../data/images/Hydrography_of_{num_epoch}_epoch_{curr_datetime_str}.png"
    )
    plt.close()
    return nse_list_basins


def plot_NSE_CDF(nse_losses, title_for_legend):
    nse_losses_np = np.array(nse_losses)
    nse_losses_np = nse_losses_np[nse_losses_np >= 0]
    # evaluate the histogram
    values, base = np.histogram(nse_losses_np, bins=100)
    # evaluate the cumulative
    cumulative = np.cumsum(values)
    cumulative = (cumulative - np.min(cumulative)) / np.max(cumulative)
    # plot the cumulative function
    plt.plot(base[:-1], cumulative, label=title_for_legend)


def read_basins_csv_files(folder_name, num_basins):
    df = pd.DataFrame(columns=["date", "precip", "flow"])
    data_csv_files = [f for f in listdir(folder_name) if isfile(join(folder_name, f))]
    for i in range(min(num_basins, len(data_csv_files))):
        data_file = data_csv_files[i]
        if str(data_file).endswith(".csv"):
            df_temp = pd.read_csv(folder_name + os.sep + data_file)
            df = pd.concat([df, df_temp])
    return df


def prepare_datasets(
    sequence_length,
    all_station_ids_train,
    all_station_ids_test,
    static_attributes_names,
    dynamic_attributes_names,
    discharge_str,
    dynamic_data_folder,
    static_data_folder,
    discharge_data_folder,
    dataset_to_use,
    create_box_plots=False,
):
    if dataset_to_use == "ERA5":
        training_data = ERA5_dataset.Dataset_ERA5(
            dynamic_data_folder=dynamic_data_folder,
            static_data_folder=static_data_folder,
            dynamic_attributes_names=dynamic_attributes_names,
            static_attributes_names=static_attributes_names,
            train_start_date="01/10/1999",
            train_end_date="30/09/2008",
            validation_start_date="01/10/1980",
            validation_end_date="30/09/1989",
            test_start_date="01/10/1989",
            test_end_date="30/09/1999",
            stage="train",
            all_stations_ids=all_station_ids_train,
            sequence_length=sequence_length,
            discharge_str=discharge_str,
        )
        test_data = ERA5_dataset.Dataset_ERA5(
            dynamic_data_folder=dynamic_data_folder,
            static_data_folder=static_data_folder,
            dynamic_attributes_names=dynamic_attributes_names,
            static_attributes_names=static_attributes_names,
            train_start_date="01/10/1999",
            train_end_date="30/09/2008",
            validation_start_date="01/10/1980",
            validation_end_date="30/09/1989",
            test_start_date="01/10/1989",
            test_end_date="30/09/1999",
            stage="validation",
            all_stations_ids=all_station_ids_test,
            sequence_length=sequence_length,
            discharge_str=discharge_str,
            x_mins=training_data.get_x_mins(),
            x_maxs=training_data.get_x_maxs(),
            y_mean=training_data.get_y_mean(),
            y_std=training_data.get_y_std(),
        )
    elif dataset_to_use == "CAMELS":
        training_data = CAMELS_dataset.Dataset_CAMELS(
            dynamic_data_folder=dynamic_data_folder,
            static_data_folder=static_data_folder,
            discharge_data_folder=discharge_data_folder,
            dynamic_attributes_names=dynamic_attributes_names,
            static_attributes_names=static_attributes_names,
            train_start_date="01/10/1999",
            train_end_date="30/09/2008",
            validation_start_date="01/10/1980",
            validation_end_date="30/09/1989",
            test_start_date="01/10/1989",
            test_end_date="30/09/1999",
            stage="train",
            all_stations_ids=all_station_ids_train,
            sequence_length=sequence_length,
            discharge_str=discharge_str,
        )
        test_data = CAMELS_dataset.Dataset_CAMELS(
            dynamic_data_folder=dynamic_data_folder,
            static_data_folder=static_data_folder,
            dynamic_attributes_names=dynamic_attributes_names,
            static_attributes_names=static_attributes_names,
            discharge_data_folder=discharge_data_folder,
            train_start_date="01/10/1999",
            train_end_date="30/09/2008",
            validation_start_date="01/10/1980",
            validation_end_date="30/09/1989",
            test_start_date="01/10/1989",
            test_end_date="30/09/1999",
            stage="validation",
            all_stations_ids=all_station_ids_test,
            sequence_length=sequence_length,
            discharge_str=discharge_str,
            x_mins=training_data.get_x_mins(),
            x_maxs=training_data.get_x_maxs(),
            y_mean=training_data.get_y_mean(),
            y_std=training_data.get_y_std(),
        )
    else:
        raise Exception(f"wrong dataset type: {dataset_to_use}")
    if create_box_plots:
        training_data.create_boxplot_of_entire_dataset()
        test_data.create_boxplot_of_entire_dataset()
        all_attributes_names = dynamic_attributes_names + static_attributes_names
        for i in range(test_data.X_data.shape[1]):
            dict_boxplots_data = {
                f"{ERA5_dataset.ATTRIBUTES_TO_TEXT_DESC[all_attributes_names[i]]}-test": test_data.X_data[
                    :, i
                ],
                f"{ERA5_dataset.ATTRIBUTES_TO_TEXT_DESC[all_attributes_names[i]]}-train": training_data.X_data[
                    :, i
                ],
            }
            ERA5_dataset.Dataset_ERA5.create_boxplot_on_data(
                dict_boxplots_data,
                plot_title=f"{ERA5_dataset.ATTRIBUTES_TO_TEXT_DESC[all_attributes_names[i]]}",
            )
    return training_data, test_data


def run_single_parameters_check_with_cross_val_on_basins(
    all_stations_list,
    sequence_length,
    learning_rate,
    num_hidden_units,
    num_epochs,
    dropout_rate,
    static_attributes,
    dynamic_attributes,
    use_Transformer,
    static_attributes_names,
    dynamic_attributes_names,
    discharge_str,
    dynamic_data_folder_train,
    static_data_folder,
    discharge_data_folder,
    dataset_to_use,
):
    split_stations_list = [
        all_stations_list[
            i : (i + math.ceil(len(all_stations_list) / K_VALUE_CROSS_VALIDATION))
        ]
        for i in range(
            0,
            len(all_stations_list),
            math.ceil(len(all_stations_list) / K_VALUE_CROSS_VALIDATION),
        )
    ]
    training_loss_list = np.zeros((K_VALUE_CROSS_VALIDATION, num_epochs))
    validation_loss_list = np.zeros((K_VALUE_CROSS_VALIDATION, num_epochs))
    nse_list_single_cross_val = []
    for i in range(len(split_stations_list)):
        train_stations_list = list(
            itertools.chain.from_iterable(
                split_stations_list[:i] + split_stations_list[i + 1 :]
            )
        )
        training_data, test_data = prepare_datasets(
            sequence_length,
            train_stations_list,
            split_stations_list[i],
            static_attributes_names,
            dynamic_attributes_names,
            discharge_str,
            dynamic_data_folder_train,
            static_data_folder,
            discharge_data_folder,
            dataset_to_use=dataset_to_use,
        )
        training_data.set_sequence_length(sequence_length)
        test_data.set_sequence_length(sequence_length)
        (
            nse_list_single_pass,
            training_loss_list_single_pass,
            validation_loss_list_single_pass,
        ) = run_training_and_test(
            learning_rate,
            sequence_length,
            num_hidden_units,
            num_epochs,
            training_data,
            test_data,
            dropout_rate,
            static_attributes,
            dynamic_attributes,
            use_Transformer=use_Transformer,
        )
        training_loss_list[i] = training_loss_list_single_pass
        validation_loss_list[i] = validation_loss_list_single_pass
        nse_list_single_cross_val.extend(nse_list_single_pass)
    plt.title(
        f"loss in {num_epochs} epochs for the parameters: "
        f"{dropout_rate};"
        f"{sequence_length};"
        f"{num_hidden_units}"
    )
    plt.plot(training_loss_list.mean(axis=0), label="training")
    plt.plot(validation_loss_list.mean(axis=0), label="validation")
    plt.legend(loc="upper left")
    plt.savefig(
        f"../data/images/training_loss_in_{num_epochs}_with_parameters: "
        f"{str(dropout_rate).replace('.', '_')};"
        f"{sequence_length};"
        f"{num_hidden_units}"
    )
    plt.close()
    return nse_list_single_cross_val


def run_single_parameters_check_with_val_on_years(
    all_stations_list,
    sequence_length,
    learning_rate,
    num_hidden_units,
    num_epochs,
    dropout_rate,
    use_Transformer,
    static_attributes_names,
    dynamic_attributes_names,
    discharge_str,
    dynamic_data_folder_train,
    static_data_folder,
    discharge_data_folder,
    dataset_to_use,
):
    training_data, test_data = prepare_datasets(
        sequence_length,
        all_stations_list[: int(0.8 * len(all_stations_list))],
        all_stations_list[int(0.8 * len(all_stations_list)) :],
        static_attributes_names,
        dynamic_attributes_names,
        discharge_str,
        dynamic_data_folder_train,
        static_data_folder,
        discharge_data_folder,
        dataset_to_use,
    )
    training_data.set_sequence_length(sequence_length)
    test_data.set_sequence_length(sequence_length)
    (
        nse_list_single_pass,
        training_loss_list_single_pass,
        validation_loss_list_single_pass,
    ) = run_training_and_test(
        learning_rate,
        sequence_length,
        num_hidden_units,
        num_epochs,
        training_data,
        test_data,
        dropout_rate,
        static_attributes_names,
        dynamic_attributes_names,
        use_Transformer=use_Transformer,
    )
    plt.title(
        f"loss in {num_epochs} epochs for the parameters: "
        f"{dropout_rate};"
        f"{sequence_length};"
        f"{num_hidden_units}"
    )
    plt.plot(training_loss_list_single_pass, label="training")
    plt.plot(validation_loss_list_single_pass, label="validation")
    plt.legend(loc="upper left")
    plt.savefig(
        f"../data/images/training_loss_in_{num_epochs}_with_parameters: "
        f"{str(dropout_rate).replace('.', '_')};"
        f"{sequence_length};"
        f"{num_hidden_units}"
    )
    plt.show()
    plt.close()
    return nse_list_single_pass


def run_training_and_test(
    learning_rate,
    sequence_length,
    num_hidden_units,
    num_epochs,
    training_data,
    test_data,
    dropout,
    static_attributes_names,
    dynamic_attributes_names,
    use_Transformer,
    calc_nse_interval=1,
):
    train_dataloader = DataLoader(
        training_data, batch_size=256, shuffle=True, num_workers=1
    )
    test_dataloader = DataLoader(
        test_data, batch_size=256, shuffle=False, num_workers=1
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if use_Transformer:
        model = FloodML_transformer.ERA5_Transformer(
            input_dim=len(dynamic_attributes_names) + len(static_attributes_names),
            sequence_length=sequence_length,
            dim_model=512,
            num_heads=8,
            num_encoder_layers=6,
            dropout_p=dropout,
        ).to(device)
    else:
        model = FloodML_lstm.FLOODML_LSTM(
            input_dim=len(dynamic_attributes_names) + len(static_attributes_names),
            hidden_dim=num_hidden_units,
            dropout=dropout,
        ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)
    loss_func = nn.MSELoss()
    loss_list_training = []
    loss_list_test = []
    nse_list = []
    preds_obs_dict_per_basin = {}
    for i in range(num_epochs):
        loss_on_training_epoch = train_epoch(
            model, optimizer, train_dataloader, loss_func, epoch=(i + 1), device=device
        )
        loss_list_training.append(loss_on_training_epoch)
        loss_on_test_epoch = eval_model(
            model, test_dataloader, device, preds_obs_dict_per_basin, loss_func
        )
        loss_list_test.append(loss_on_test_epoch)
        if (i % calc_nse_interval) == (calc_nse_interval - 1):
            nse_list_epoch = calc_validation_basins_nse(
                preds_obs_dict_per_basin, (i + 1)
            )
            preds_obs_dict_per_basin.clear()
            nse_list.extend(nse_list_epoch)
    if len(nse_list) > 0:
        print(
            f"parameters are: dropout={dropout} sequence_length={sequence_length} "
            f"num_hidden_units={num_hidden_units} num_epochs={num_epochs}, median NSE is: {statistics.median(nse_list)}"
        )
    return nse_list, loss_list_training, loss_list_test


def choose_hyper_parameters_validation(
    static_attributes_names,
    dynamic_attributes_names,
    discharge_str,
    dynamic_data_folder_train,
    static_data_folder,
    discharge_data_folder,
    use_transformer,
    dataset_to_use,
):
    all_stations_list = (
        open("../data/CAMELS_US/train_basins.txt", "r").read().splitlines()
    )
    shuffle(all_stations_list)
    learning_rates = np.linspace(5 * (10 ** -3), 5 * (10 ** -3), num=1).tolist()
    dropout_rates = [0.0, 0.25, 0.4, 0.5]
    sequence_lengths = [30, 90, 180, 270, 365]
    if use_transformer:
        num_hidden_units = [1]
    else:
        num_hidden_units = [96, 128, 156, 196, 224, 64, 256]
    num_epochs = [10]
    dict_results = {
        "learning rate": [],
        "sequence length": [],
        "num epochs": [],
        "num hidden units": [],
        "median NSE": [],
    }
    best_median_nse = -1
    list_nse_lists_basins = []
    list_plots_titles = []
    all_parameters = list(
        itertools.product(
            learning_rates,
            dropout_rates,
            sequence_lengths,
            num_hidden_units,
            num_epochs,
        )
    )
    curr_datetime = datetime.now()
    curr_datetime_str = curr_datetime.strftime("%d-%m-%Y_%H:%M:%S")
    for (
        learning_rate_param,
        dropout_rate_param,
        sequence_length_param,
        num_hidden_units_param,
        num_epochs_param,
    ) in all_parameters:
        nse_list = []
        nse_list_single_pass = run_single_parameters_check_with_val_on_years(
            all_stations_list,
            sequence_length_param,
            learning_rate_param,
            num_hidden_units_param,
            num_epochs_param,
            dropout_rate_param,
            use_transformer,
            static_attributes_names,
            dynamic_attributes_names,
            discharge_str,
            dynamic_data_folder_train,
            static_data_folder,
            discharge_data_folder,
            dataset_to_use=dataset_to_use,
        )
        nse_list.extend(nse_list_single_pass)
        if len(nse_list) == 0:
            median_nse = -1
        else:
            median_nse = statistics.median(nse_list)
        if len(nse_list) > 0:
            list_plots_titles.append(
                f"{learning_rate_param};"
                f"{sequence_length_param};"
                f"{num_hidden_units_param};"
                f"{num_epochs_param}"
            )
            list_nse_lists_basins.append(nse_list)
        if median_nse > best_median_nse or best_median_nse == -1:
            best_median_nse = median_nse
            best_parameters = (
                learning_rate_param,
                sequence_length_param,
                num_hidden_units_param,
                num_epochs_param,
            )
        dict_results["learning rate"].append(learning_rate_param)
        dict_results["sequence length"].append(sequence_length_param)
        dict_results["num hidden units"].append(num_hidden_units_param)
        dict_results["num epochs"].append(num_epochs_param)
        dict_results["median NSE"].append(median_nse)
        for list_nse, title in zip(list_nse_lists_basins, list_plots_titles):
            plot_NSE_CDF(list_nse, title)
        plt.grid()
        plt.legend()
        plt.savefig(
            "../data/images/parameters_comparison" + f"_{curr_datetime_str}" + ".png"
        )
        plt.close()
        df_results = pd.DataFrame(data=dict_results)
        df_results.to_csv(
            f"./results_{curr_datetime_str}.csv",
            mode="a",
            header=not os.path.exists(f"./results_{curr_datetime_str}.csv"),
        )
        print(f"best parameters: {best_parameters}")
    return best_parameters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        help="which dataset to train and test on - the options are CAMELS or ERA5",
        choices=["CAMELS", "ERA5"],
        default="ERA5",
    )
    parser.add_argument(
        "--model",
        help="which model to use - the options are LSTM or Transformer",
        choices=["LSTM", "Transformer"],
        default="LSTM",
    )
    command_args = parser.parse_args()
    use_Transformer = command_args.model == "Transformer"
    print(
        f"running with dataset: {command_args.dataset} and with model: {command_args.model}"
    )
    if command_args.dataset == "CAMELS":
        choose_hyper_parameters_validation(
            CAMELS_dataset.STATIC_ATTRIBUTES_NAMES,
            CAMELS_dataset.DYNAMIC_ATTRIBUTES_NAMES,
            CAMELS_dataset.DISCHARGE_STR,
            CAMELS_dataset.DYNAMIC_DATA_FOLDER,
            CAMELS_dataset.STATIC_DATA_FOLDER,
            CAMELS_dataset.DISCHARGE_DATA_FOLDER,
            use_transformer=use_Transformer,
            dataset_to_use="CAMELS",
        )
    elif command_args.dataset == "ERA5":
        choose_hyper_parameters_validation(
            ERA5_dataset.STATIC_ATTRIBUTES_NAMES,
            ERA5_dataset.DYNAMIC_ATTRIBUTES_NAMES_CARAVAN,
            ERA5_dataset.DISCHARGE_STR_CARAVAN,
            ERA5_dataset.DYNAMIC_DATA_FOLDER_CARAVAN,
            ERA5_dataset.STATIC_DATA_FOLDER,
            ERA5_dataset.DISCHARGE_DATA_FOLDER_CARAVAN,
            use_transformer=use_Transformer,
            dataset_to_use="ERA5",
        )
    else:
        raise Exception(f"wrong dataset name: {command_args.dataset}")


if __name__ == "__main__":
    sys.argv = ["", "--model", "Transformer", "--dataset", "ERA5"]
    main()
