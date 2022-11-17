import os
import sys
from torch.utils.data import DataLoader
from ERA5_dataset import Dataset_ERA5
from ERA5_dataset import ATTRIBUTES_TO_TEXT_DESC
from tqdm import tqdm
from ERA5_lstm import ERA5_LSTM
import torch.optim
import torch.nn as nn
from os import listdir
from os.path import isfile, join
import pandas as pd
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from typing import Tuple
import numpy as np
import itertools
from datetime import datetime
import statistics
from random import shuffle

K_VALUE_CROSS_VALIDATION = 4


def eval_model(model, loader, device, preds_obs_dict_per_basin) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the model.

    :param model: A torch.nn.Module implementing the LSTM model
    :param loader: A PyTorch DataLoader, providing the data.

    :return: Two torch Tensors, containing the observations and
        model predictions

    Parameters
    ----------
    preds_obs_dict_per_basin
    """
    # set model to eval mode (important for dropout)
    model.eval()
    # in inference mode, we don't need to store intermediate steps for
    # backprob
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
            preds_obs_dict_per_basin[station_id].append((ys, y_hat))


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


def calc_validation_basins_nse(preds_obs_dict_per_basin, num_epoch, num_basins_for_nse_calc=10):
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
        print(f"station with id: {stations_id} has nse of: {nse}")
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
    plt.savefig(f"../data/images/Hydrography_of_{num_epoch}_epoch_{curr_datetime_str}.png")
    plt.close()
    return nse_list_basins


def train_epoch(model, optimizer, loader, loss_func, epoch, device):
    # # set model to train mode (important for dropout)
    model.train()
    pbar = tqdm(loader, file=sys.stdout)
    print(f"Epoch {epoch}")
    pbar.set_description(f"Epoch {epoch}")
    loss_list = []
    # request mini-batch of data from the loader
    running_loss = 0.0
    i = 0
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
        # update the weights
        optimizer.step()
        # write current loss in the progress bar
        running_loss += loss.item()
        if i % 200 == 199:  # print every 2000 mini-batches
            print(f'[{epoch}] loss: {running_loss / 200:.3f}')
            loss_list.append(running_loss / 200)
            running_loss = 0.0
        print(f"Loss: {loss.item():.4f}")
        pbar.set_postfix_str(f"Loss: {loss.item():.4f}")
        i += 1
    return loss_list


def plot_NSE_CDF(nse_losses, title_for_legend):
    nse_losses_np = np.array(nse_losses)
    nse_losses_np = nse_losses_np[nse_losses_np >= 0]
    # taken from https://stackoverflow.com/questions/15408371/cumulative-distribution-plots-python
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


def prepare_training_data_and_test_data(sequence_length,
                                        all_station_files_train,
                                        all_station_files_test,
                                        static_attributes_names,
                                        dynamic_attributes_names,
                                        discharge_str,
                                        dynamic_data_folder_train,
                                        dynamic_data_folder_test,
                                        use_Caravan_dataset=True,
                                        create_boxplots=False):
    training_data = Dataset_ERA5(dynamic_data_folder=dynamic_data_folder_train,
                                 static_data_file_caravan="../data/ERA5/Caravan/attributes/attributes_caravan_us.csv",
                                 static_data_file_hydroatlas="../data/ERA5/Caravan/attributes"
                                                             "/attributes_hydroatlas_us.csv",
                                 dynamic_attributes_names=dynamic_attributes_names,
                                 discharge_str=discharge_str,
                                 static_attributes_names=static_attributes_names,
                                 train_start_date='01/10/1999',
                                 train_end_date='30/09/2008',
                                 validation_start_date='01/10/1989',
                                 validation_end_date='30/09/1999',
                                 test_start_date='01/10/1989',
                                 test_end_date='30/09/1999',
                                 stage="train",
                                 all_station_files=all_station_files_train,
                                 sequence_length=sequence_length,
                                 use_Caravan_dataset=use_Caravan_dataset)
    test_data = Dataset_ERA5(dynamic_data_folder=dynamic_data_folder_test,
                             static_data_file_caravan="../data/ERA5/Caravan/attributes/attributes_caravan_us.csv",
                             static_data_file_hydroatlas="../data/ERA5/Caravan/attributes"
                                                         "/attributes_hydroatlas_us.csv",
                             dynamic_attributes_names=dynamic_attributes_names,
                             discharge_str=discharge_str,
                             static_attributes_names=static_attributes_names,
                             train_start_date='01/10/1999',
                             train_end_date='30/09/2008',
                             validation_start_date='01/10/1989',
                             validation_end_date='30/09/1999',
                             test_start_date='01/10/1989',
                             test_end_date='30/09/1999',
                             stage="test",
                             all_station_files=all_station_files_test,
                             x_means=training_data.get_x_mean(),
                             x_stds=training_data.get_x_std(),
                             y_mean=training_data.get_y_mean(),
                             y_std=training_data.get_y_std(),
                             sequence_length=sequence_length)
    if create_boxplots:
        training_data.create_boxplot_of_entire_dataset()
        test_data.create_boxplot_of_entire_dataset()
        all_attributes_names = dynamic_attributes_names + static_attributes_names
        for i in range(test_data.X_data.shape[1]):
            dict_boxplots_data = {f"{ATTRIBUTES_TO_TEXT_DESC[all_attributes_names[i]]}-test": test_data.X_data[:, i],
                                  f"{ATTRIBUTES_TO_TEXT_DESC[all_attributes_names[i]]}-train": training_data.X_data[:, i]}
            Dataset_ERA5.create_boxplot_on_data(dict_boxplots_data,
                                                plot_title=f"{ATTRIBUTES_TO_TEXT_DESC[all_attributes_names[i]]}")
    return training_data, test_data


def run_training_and_test(learning_rate, sequence_length,
                          num_hidden_units,
                          num_epochs, training_data,
                          test_data, calc_nse_interval=1):
    use_Caravan_dataset = True
    static_attributes_names = ["ele_mt_sav", "slp_dg_sav", "basin_area", "for_pc_sse",
                               "cly_pc_sav", "slt_pc_sav", "snd_pc_sav", "soc_th_sav",
                               "p_mean", "pet_mean",
                               "aridity", "frac_snow",
                               "high_prec_freq",
                               "high_prec_dur",
                               "low_prec_freq", "low_prec_dur"]
    if use_Caravan_dataset:
        dynamic_attributes_names = ["total_precipitation_sum", "temperature_2m_min",
                                    "temperature_2m_max", "potential_evaporation_sum",
                                    "surface_net_solar_radiation_mean"]
    else:
        dynamic_attributes_names = ["precip"]
    train_dataloader = DataLoader(training_data, batch_size=128, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ERA5_LSTM(hidden_dim=num_hidden_units,
                      input_dim=len(static_attributes_names) + len(dynamic_attributes_names)).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    loss_func = nn.MSELoss()
    loss_list = []
    nse_list = []
    preds_obs_dict_per_basin = {}
    for i in range(num_epochs):
        loss_list_epoch = train_epoch(model, optimizer, train_dataloader, loss_func, epoch=(i + 1), device=device)
        loss_list.extend(loss_list_epoch)
        eval_model(model, test_dataloader, device, preds_obs_dict_per_basin)
        if (i % calc_nse_interval) == (calc_nse_interval - 1):
            nse_list_epoch = calc_validation_basins_nse(preds_obs_dict_per_basin, (i + 1))
            preds_obs_dict_per_basin.clear()
            nse_list.extend(nse_list_epoch)
    plt.title(f"loss in {num_epochs} epochs for the parameters: {learning_rate};{sequence_length};{num_hidden_units}")
    plt.plot(loss_list)
    plt.savefig(f"../data/images/loss_in_{num_epochs}_with_parameters: "
                f"{str(learning_rate).replace('.', '_')};{sequence_length};{num_hidden_units}")
    plt.close()
    if len(nse_list) > 0:
        print(f"parameters are: learning_rate={learning_rate} sequence_length={sequence_length} "
              f"num_hidden_units={num_hidden_units} num_epochs={num_epochs}, median NSE is: {statistics.median(nse_list)}")
    return nse_list


def choose_hyper_parameters_validation(static_attributes_names,
                                       dynamic_attributes_names,
                                       discharge_str,
                                       dynamic_data_folder_train,
                                       dynamic_data_folder_test,
                                       use_Caravan_dataset):
    learning_rates = np.linspace(10 ** -5, 10 ** -5, num=1).tolist()
    dropout_rates = [0.0, 0.25, 0.4, 0.5]
    sequence_lengths = [30, 90, 180, 270, 365]
    num_hidden_units = [64, 96, 128, 156, 196, 224, 256]
    num_epochs = [5]
    dict_results = {"learning rate": [], "sequence length": [], "num epochs": [], "num hidden units": [],
                    "median NSE": []}
    best_median_nse = -1
    list_nse_lists_basins = []
    list_plots_titles = []
    all_parameters = list(itertools.product(learning_rates, sequence_lengths, num_hidden_units, num_epochs))
    curr_datetime = datetime.now()
    curr_datetime_str = curr_datetime.strftime("%d-%m-%Y_%H:%M:%S")
    for learning_rate_param, sequence_length_param, num_hidden_units_param, num_epochs_param in all_parameters:
        all_stations_for_validation = [f for f in listdir(dynamic_data_folder_train) if
                                       isfile(join(dynamic_data_folder_train, f))]
        shuffle(all_stations_for_validation)
        split_stations_list = [all_stations_for_validation[i:i + len(all_stations_for_validation) // K_VALUE_CROSS_VALIDATION]
                               for i in
                               range(0, len(all_stations_for_validation), len(all_stations_for_validation) // K_VALUE_CROSS_VALIDATION)]
        nse_list = []
        for i in range(len(split_stations_list)):
            train_stations_list = list(itertools.chain.from_iterable(split_stations_list[:i] + split_stations_list[i + 1:]))
            training_data, test_data = prepare_training_data_and_test_data(sequence_length_param,
                                                                           train_stations_list,
                                                                           split_stations_list[i],
                                                                           static_attributes_names,
                                                                           dynamic_attributes_names,
                                                                           discharge_str,
                                                                           dynamic_data_folder_train,
                                                                           dynamic_data_folder_train,
                                                                           use_Caravan_dataset)
            training_data.set_sequence_length(sequence_length_param)
            test_data.set_sequence_length(sequence_length_param)
            nse_list.extend(run_training_and_test(learning_rate_param, sequence_length_param, num_hidden_units_param,
                                                  num_epochs_param, training_data, test_data))
        if len(nse_list) == 0:
            median_nse = -1
        else:
            median_nse = statistics.median(nse_list)
        if len(nse_list) > 0:
            list_plots_titles.append(f"{learning_rate_param};"
                                     f"{sequence_length_param};"
                                     f"{num_hidden_units_param};"
                                     f"{num_epochs_param}")
            list_nse_lists_basins.append(nse_list)
        if median_nse > best_median_nse or best_median_nse == -1:
            best_median_nse = median_nse
            best_parameters = (learning_rate_param,
                               sequence_length_param,
                               num_hidden_units_param,
                               num_epochs_param)
        dict_results["learning rate"].append(learning_rate_param)
        dict_results["sequence length"].append(sequence_length_param)
        dict_results["num hidden units"].append(num_hidden_units_param)
        dict_results["num epochs"].append(num_epochs_param)
        dict_results["median NSE"].append(median_nse)
        for list_nse, title in zip(list_nse_lists_basins, list_plots_titles):
            plot_NSE_CDF(list_nse, title)
        plt.grid()
        plt.legend()
        plt.savefig("../data/images/parameters_comparison" +
                    f"_{curr_datetime_str}" + ".png")
        plt.close()
        df_results = pd.DataFrame(data=dict_results)
        df_results.to_csv(f"./results_{curr_datetime_str}.csv", mode='a',
                          header=not os.path.exists(f"./results_{curr_datetime_str}.csv"))
        print(f"best parameters: {best_parameters}")
    return best_parameters


def main():
    use_Caravan_dataset = True
    static_attributes_names = ["ele_mt_sav", "slp_dg_sav", "basin_area", "for_pc_sse",
                               "cly_pc_sav", "slt_pc_sav", "snd_pc_sav", "soc_th_sav",
                               "p_mean", "pet_mean",
                               "aridity", "frac_snow",
                               "high_prec_freq",
                               "high_prec_dur",
                               "low_prec_freq", "low_prec_dur"]
    if use_Caravan_dataset:
        dynamic_attributes_names = ["total_precipitation_sum", "temperature_2m_min",
                                    "temperature_2m_max", "potential_evaporation_sum",
                                    "surface_net_solar_radiation_mean"]
        discharge_str = "streamflow"
        dynamic_data_folder_train = "../data/ERA5/Caravan/timeseries/csv/us/train/"
        dynamic_data_folder_test = "../data/ERA5/Caravan/timeseries/csv/us/test/"
    else:
        dynamic_attributes_names = ["precip"]
        discharge_str = "flow"
        dynamic_data_folder_train = "../data/ERA5/all_data_daily/train/"
        dynamic_data_folder_test = "../data/ERA5/all_data_daily/test/"

    choose_hyper_parameters_validation(static_attributes_names,
                                       dynamic_attributes_names,
                                       discharge_str,
                                       dynamic_data_folder_train,
                                       dynamic_data_folder_test,
                                       use_Caravan_dataset)


if __name__ == "__main__":
    main()
