import os
import sys
from torch.utils.data import DataLoader
import ERA5_dataset
import CAMELS_dataset
from tqdm import tqdm
import torch.optim
from os import listdir
from os.path import isfile, join
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from typing import Tuple
import numpy as np
import itertools
from datetime import datetime
import statistics
import argparse
from FloodML_LSTM import LSTM
from FloodML_Transformer_LSTM import ERA5_Transformer_LSTM
from FloodML_2_LSTM_Conv_LSTM import TWO_LSTM_CONV_LSTM
from FloodML_2_LSTM_CNN_LSTM import TWO_LSTM_CNN_LSTM
from FloodML_Transformer_Seq2Seq import Transformer_Seq2Seq
from pathlib import Path
import random
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import psutil
from torch.profiler import profile, record_function, ProfilerActivity

# wandb.login(key="ed527efc0923927fda63686bf828192a102daa48")
#
# wandb.init(project="FloodML", entity="r999")

matplotlib.use("AGG")

K_VALUE_CROSS_VALIDATION = 2

torch.multiprocessing.set_sharing_strategy('file_system')


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '10006'
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def aggregate_gradients(model, world_size):
    if world_size > 1:
        for param in model.parameters():
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= world_size


def train_epoch(model, optimizer, loader, loss_func, epoch, device, print_tqdm_to_console,
                specific_model_type):
    # set model to train mode (important for dropout)
    torch.cuda.empty_cache()
    model.train()
    if print_tqdm_to_console:
        pbar = tqdm(loader, file=sys.stdout)
    else:
        pbar = tqdm(loader, file=open('../tqdm_progress.txt', 'w'))
    pbar.set_description(f"Epoch {epoch}")
    # request mini-batch of data from the loader
    running_loss = 0.0
    for stds, station_id_batch, xs_non_spatial, xs_spatial, ys in pbar:
        xs_non_spatial, ys = xs_non_spatial.to(device), ys.to(device)
        optimizer.zero_grad()
        if xs_spatial.nelement() > 0:
            y_hat = model(xs_non_spatial, xs_spatial.to(device))
        elif specific_model_type.lower() == "transformer_seq2seq":
            ys_prefix_random = torch.cat([torch.zeros(size=(ys.shape[0], 1), device="cuda"), ys[:, :-1]], axis=-1)
            y_hat = model(xs_non_spatial, ys_prefix_random)[:, -1, :]
            ys = ys[:, -1]
        else:
            y_hat = model(xs_non_spatial)
        loss = loss_func(ys, y_hat.squeeze(0), stds.to(device).reshape(-1, 1))
        loss.backward()
        # aggregate_gradients(model, world_size)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        optimizer.step()
        running_loss += loss.item()
        # print(f"Loss: {loss.item():.4f}")
        pbar.set_postfix_str(f"Loss: {loss.item():.4f}")
    print(f"Loss on the entire training epoch: {running_loss / (len(loader)):.4f}")
    # wandb.log({"loss": running_loss / (len(loader))})
    return running_loss / (len(loader))


def eval_model(model, loader, device, epoch, print_tqdm_to_console,
               specific_model_type) -> Tuple[torch.Tensor, torch.Tensor]:
    torch.cuda.empty_cache()
    preds_obs_dict_per_basin = {}
    # set model to eval mode (important for dropout)
    model.eval()
    if print_tqdm_to_console:
        pbar = tqdm(loader, file=sys.stdout)
    else:
        pbar = tqdm(loader, file=open('../tqdm_progress.txt', 'a'))
    pbar.set_description(f"Epoch {epoch}")
    # in inference mode, we don't need to store intermediate steps for backprob
    with torch.no_grad():
        # request mini-batch of data from the loader
        for _, station_id_batch, xs_non_spatial, xs_spatial, ys in pbar:
            # push data to GPU (if available)
            xs_non_spatial, ys = xs_non_spatial.to(device), ys.to(device)
            # get model predictions
            if xs_spatial.nelement() > 0:
                y_hat = model(xs_non_spatial, xs_spatial.to(device)).squeeze()
            elif specific_model_type.lower() == "transformer_seq2seq":
                y_hat_prev = torch.zeros(size=(ys.shape[0], 1), device="cuda")
                for _ in range(xs_non_spatial.shape[1]):
                    y_hat = model(xs_non_spatial, y_hat_prev)[:, -1, :]
                    y_hat_prev = y_hat
                y_hat = y_hat.squeeze()
                ys = ys[:, -1]
            else:
                y_hat = model(xs_non_spatial).squeeze()
            pred_actual = (
                    (y_hat * loader.dataset.y_std) + loader.dataset.y_mean)
            pred_expected = (
                    (ys * loader.dataset.y_std) + loader.dataset.y_mean)
            # print(torch.cat([y_hat.cpu(), ys], dim=1))
            for i in range(len(station_id_batch)):
                station_id = station_id_batch[i]
                if station_id not in preds_obs_dict_per_basin:
                    preds_obs_dict_per_basin[station_id] = []
                preds_obs_dict_per_basin[station_id].append(
                    (pred_expected[i].clone().item(), pred_actual[i].item()))
    return preds_obs_dict_per_basin


def calc_nse_star(obs, sim, stds):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mask = ~torch.isnan(obs)
    y_hat = sim.squeeze() * mask.int().float()
    y = obs * mask.int().float()
    per_basin_target_stds = stds[torch.all(mask, dim=0)]
    squared_error = (y_hat - y) ** 2
    weights = 1 / (per_basin_target_stds + 0.1) ** 2
    scaled_loss = weights.squeeze() * squared_error
    return torch.mean(scaled_loss)


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
    mask = ~np.isnan(obs)
    sim = sim[mask]
    obs = obs[mask]
    denominator = ((obs - obs.mean()) ** 2).sum()
    numerator = ((sim - obs) ** 2).sum()
    nse_val = 1 - numerator / denominator
    return float(nse_val)


def calc_validation_basins_nse(preds_obs_dict_per_basin, num_epoch, num_basins_for_nse_calc=10):
    stations_ids = list(preds_obs_dict_per_basin.keys())
    nse_list_basins = []
    for stations_id in stations_ids:
        obs_and_preds = preds_obs_dict_per_basin[stations_id]
        obs, preds = zip(*obs_and_preds)
        obs = np.stack(list(obs))
        preds = np.stack(list(preds))
        nse = calc_nse(obs, preds)
        print(f"station with id: {stations_id} has nse of: {nse}", flush=True)
        nse_list_basins.append(nse)
    # nse_list_basins = torch.cat(nse_list_basins).cpu().numpy()
    nse_list_basins_idx_sorted = np.argsort(np.array(nse_list_basins))
    median_nse_basin = stations_ids[nse_list_basins_idx_sorted[len(stations_ids) // 2]]
    median_nse = statistics.median(nse_list_basins)
    print(f"Basin {median_nse_basin} - NSE: {median_nse:.3f}", flush=True)
    fig, ax = plt.subplots(figsize=(20, 6))
    obs_and_preds = preds_obs_dict_per_basin[median_nse_basin]
    obs, preds = zip(*obs_and_preds)
    obs = np.stack(list(obs))
    preds = np.stack(list(preds))
    ax.plot(obs.squeeze(), label="observation")
    ax.plot(preds.squeeze(), label="prediction")
    ax.legend()
    ax.set_title(f"Basin {median_nse_basin} - NSE: {median_nse:.3f}")
    plt.savefig(
        f"../data/results/Hydrograph_of_basin_{median_nse_basin}_in_epoch_{num_epoch}.png"
    )
    plt.close()
    return nse_list_basins, median_nse


def plot_NSE_CDF(nse_losses, title_for_legend):
    nse_losses_np = np.array(nse_losses)
    nse_losses_np = nse_losses_np[nse_losses_np >= 0]
    # evaluate the histogram
    values, base = np.histogram(nse_losses_np, bins=100)
    # evaluate the cumulative
    cumulative = np.cumsum(values)
    cumulative = (cumulative - np.min(cumulative)) / np.max(cumulative)
    # plot the cumulative function
    plt.title(f"CDF of NSE of basins - {title_for_legend}")
    plt.xlabel("NSE")
    plt.ylabel("CDF")
    plt.plot(base[:-1], cumulative)


def read_basins_csv_files(folder_name, num_basins):
    df = pd.DataFrame(columns=["date", "precip", "flow"])
    data_csv_files = [f for f in listdir(folder_name) if isfile(join(folder_name, f))]
    for i in range(min(num_basins, len(data_csv_files))):
        data_file = data_csv_files[i]
        if str(data_file).endswith(".csv"):
            df_temp = pd.read_csv(folder_name + os.sep + data_file)
            df = pd.concat([df, df_temp])
    return df


def sort_basins_by_static_attributes(static_data_folder):
    df_attr_caravan = pd.read_csv(
        Path(static_data_folder) / "attributes_hydroatlas_us.csv",
        dtype={"gauge_id": str},
    )
    df_attr_hydroatlas = pd.read_csv(
        Path(static_data_folder) / "attributes_caravan_us.csv",
        dtype={"gauge_id": str},
    )
    df_attr = df_attr_caravan.merge(df_attr_hydroatlas, on="gauge_id")
    df_attr["gauge_id"] = (
        df_attr["gauge_id"]
        .apply(lambda x: str(x).replace("us_", ""))
        .values.tolist()
    )
    df_attr = df_attr.dropna()
    # df_attr = df_attr[["gauge_id", "basin_area"]].sort_values(["basin_area"], ascending=False)
    return df_attr["gauge_id"].to_list()


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
        specific_model_type,
        sequence_length_spatial,
        create_box_plots=False,
        create_new_files=False,
        limit_size_above_1000=False,
        use_all_static_attr=False
):
    print(f"running with dataset: {dataset_to_use}")
    if dataset_to_use == "ERA5" or dataset_to_use == "CARAVAN":
        use_Caravan_dataset = True if dataset_to_use == "CARAVAN" else False
        training_data = ERA5_dataset.Dataset_ERA5(
            main_folder=ERA5_dataset.MAIN_FOLDER,
            dynamic_data_folder=dynamic_data_folder,
            static_data_folder=static_data_folder,
            dynamic_attributes_names=dynamic_attributes_names,
            static_attributes_names=static_attributes_names,
            train_start_date="01/10/1997",
            train_end_date="30/09/2007",
            validation_start_date="01/10/1988",
            validation_end_date="30/09/1997",
            test_start_date="01/10/1988",
            test_end_date="30/09/1997",
            stage="train",
            specific_model_type=specific_model_type,
            all_stations_ids=all_station_ids_train,
            sequence_length=sequence_length,
            discharge_str=discharge_str,
            use_Caravan_dataset=use_Caravan_dataset,
            create_new_files=create_new_files,
            sequence_length_spatial=sequence_length_spatial,
            limit_size_above_1000=limit_size_above_1000,
            use_all_static_attr=use_all_static_attr
        )
        test_data = ERA5_dataset.Dataset_ERA5(
            main_folder=ERA5_dataset.MAIN_FOLDER,
            dynamic_data_folder=dynamic_data_folder,
            static_data_folder=static_data_folder,
            dynamic_attributes_names=dynamic_attributes_names,
            static_attributes_names=static_attributes_names,
            train_start_date="01/10/1997",
            train_end_date="30/09/2007",
            validation_start_date="01/10/1988",
            validation_end_date="30/09/1997",
            test_start_date="01/10/1988",
            test_end_date="30/09/1997",
            stage="validation",
            all_stations_ids=all_station_ids_test,
            sequence_length=sequence_length,
            discharge_str=discharge_str,
            specific_model_type=specific_model_type,
            use_Caravan_dataset=use_Caravan_dataset,
            y_std=training_data.y_std,
            y_mean=training_data.y_mean,
            x_means=training_data.x_means,
            x_stds=training_data.x_stds,
            create_new_files=create_new_files,
            sequence_length_spatial=sequence_length_spatial,
            limit_size_above_1000=limit_size_above_1000,
            use_all_static_attr=use_all_static_attr
        )
    elif dataset_to_use == "CAMELS":
        training_data = CAMELS_dataset.Dataset_CAMELS(
            main_folder=CAMELS_dataset.MAIN_FOLDER,
            dynamic_data_folder=dynamic_data_folder,
            static_data_folder=static_data_folder,
            discharge_data_folder=discharge_data_folder,
            dynamic_attributes_names=dynamic_attributes_names,
            static_attributes_names=static_attributes_names,
            train_start_date="01/10/1997",
            train_end_date="30/09/2007",
            validation_start_date="01/10/1988",
            validation_end_date="30/09/1997",
            test_start_date="01/10/1988",
            test_end_date="30/09/1997",
            stage="train",
            specific_model_type=specific_model_type,
            sequence_length_spatial=sequence_length_spatial,
            create_new_files=create_new_files,
            all_stations_ids=all_station_ids_train,
            sequence_length=sequence_length,
            discharge_str=discharge_str,
            use_all_static_attr=use_all_static_attr,
            limit_size_above_1000=limit_size_above_1000
        )
        test_data = CAMELS_dataset.Dataset_CAMELS(
            main_folder=CAMELS_dataset.MAIN_FOLDER,
            dynamic_data_folder=dynamic_data_folder,
            static_data_folder=static_data_folder,
            dynamic_attributes_names=dynamic_attributes_names,
            static_attributes_names=static_attributes_names,
            discharge_data_folder=discharge_data_folder,
            train_start_date="01/10/1997",
            train_end_date="30/09/2007",
            validation_start_date="01/10/1988",
            validation_end_date="30/09/1997",
            test_start_date="01/10/1988",
            test_end_date="30/09/1997",
            stage="validation",
            specific_model_type=specific_model_type,
            sequence_length_spatial=sequence_length_spatial,
            create_new_files=create_new_files,
            all_stations_ids=all_station_ids_test,
            sequence_length=sequence_length,
            discharge_str=discharge_str,
            y_std=training_data.y_std,
            y_mean=training_data.y_mean,
            x_means=training_data.x_means,
            x_stds=training_data.x_stds,
            use_all_static_attr=use_all_static_attr,
            limit_size_above_1000=limit_size_above_1000
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
    print('RAM Used (GB):', psutil.virtual_memory()[3] / 1000000000)
    return training_data, test_data


def run_single_parameters_check_with_val_on_years(
        train_stations_list,
        val_stations_list,
        sequence_length,
        learning_rate,
        num_hidden_units,
        dropout_rate,
        static_attributes_names,
        dynamic_attributes_names,
        discharge_str,
        dynamic_data_folder_train,
        static_data_folder,
        discharge_data_folder,
        model_name,
        dataset_to_use,
        optim_name,
        shared_model,
        num_epochs,
        num_workers_data_loader,
        profile_code,
        num_processes_ddp,
        create_new_files,
        sequence_length_spatial,
        print_tqdm_to_console,
        limit_size_above_1000,
        use_all_static_attr
):
    print(f"number of workers using for data loader is: {num_workers_data_loader}")
    specific_model_type = "CONV" if "CONV" in model_name else "CNN" if "CNN" in model_name else \
        "Transformer_Seq2Seq" if "Transformer_Seq2Seq" in model_name else "Transformer_LSTM" if \
            "Transformer_LSTM" in model_name else "LSTM"
    training_data, test_data = prepare_datasets(
        sequence_length,
        train_stations_list,
        val_stations_list,
        static_attributes_names,
        dynamic_attributes_names,
        discharge_str,
        dynamic_data_folder_train,
        static_data_folder,
        discharge_data_folder,
        dataset_to_use,
        sequence_length_spatial=sequence_length_spatial,
        specific_model_type=specific_model_type,
        create_new_files=create_new_files,
        limit_size_above_1000=limit_size_above_1000,
        use_all_static_attr=use_all_static_attr
    )
    training_data.set_sequence_length(sequence_length)
    test_data.set_sequence_length(sequence_length)
    nse_list_last_epoch, training_loss_list_single_pass = \
        run_training_and_test(learning_rate,
                              sequence_length,
                              num_hidden_units,
                              num_epochs,
                              training_data,
                              test_data,
                              dropout_rate,
                              static_attributes_names,
                              dynamic_attributes_names,
                              model_name,
                              1,
                              optim_name,
                              num_workers_data_loader,
                              profile_code,
                              sequence_length_spatial,
                              print_tqdm_to_console,
                              specific_model_type
                              )
    if len(nse_list_last_epoch) > 0:
        print(
            f"parameters are: dropout={dropout_rate} sequence_length={sequence_length} "
            f"num_hidden_units={num_hidden_units} num_epochs={num_epochs}, median NSE is: "
            f"{statistics.median(nse_list_last_epoch)}"
        )
    plt.title(
        f"loss in {num_epochs} epochs for the parameters: "
        f"{dropout_rate};"
        f"{sequence_length};"
        f"{num_hidden_units}"
    )
    plt.plot(training_loss_list_single_pass, label="training")
    plt.legend(loc="upper left")
    plt.savefig(
        f"../data/results/training_loss_in_{num_epochs}_with_parameters: "
        f"{str(dropout_rate).replace('.', '_')};"
        f"{sequence_length};"
        f"{num_hidden_units}"
    )
    plt.show()
    plt.close()
    return nse_list_last_epoch


class DistributedSamplerNoDuplicate(DistributedSampler):
    """ A distributed sampler that doesn't add duplicates. Arguments are the same as DistributedSampler """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.drop_last and len(self.dataset) % self.num_replicas != 0:
            # some ranks may have less samples, that's fine
            if self.rank >= len(self.dataset) % self.num_replicas:
                self.num_samples -= 1
            self.total_size = len(self.dataset)


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
        model_name,
        calc_nse_interval,
        optim_name,
        num_workers_data_loader,
        profile_code,
        sequence_length_spatial,
        print_tqdm_to_console,
        specific_model_type
):
    print('RAM Used (GB):', psutil.virtual_memory()[3] / 1000000000)
    print(f"running with model: {model_name}")
    if model_name.lower() == "transformer_lstm":
        model = ERA5_Transformer_LSTM(sequence_length=sequence_length,
                                      num_in_features_encoder=len(dynamic_attributes_names)
                                                              + len(training_data.list_static_attributes_names),
                                      num_hidden_lstm=num_hidden_units,
                                      dropout_rate=dropout,
                                      num_out_features_encoder=128)
    elif model_name.lower() == "transformer_seq2seq":
        model = Transformer_Seq2Seq(
            in_features=len(dynamic_attributes_names) + len(training_data.list_static_attributes_names))
    elif model_name.lower() == "conv_lstm":
        model = TWO_LSTM_CONV_LSTM(dropout=dropout,
                                   input_dim=len(dynamic_attributes_names) + len(
                                       training_data.list_static_attributes_names),
                                   hidden_dim=num_hidden_units, sequence_length_conv_lstm=sequence_length_spatial,
                                   in_channels_cnn=1, image_width=training_data.max_dim,
                                   image_height=training_data.max_dim)
    elif model_name.lower() == "lstm":
        model = LSTM(
            num_static_attr=len(training_data.list_static_attributes_names),
            num_dynamic_attr=len(dynamic_attributes_names),
            hidden_dim=num_hidden_units,
            dropout=dropout)
    elif model_name.lower() == "cnn_lstm":
        model = TWO_LSTM_CNN_LSTM(
            input_dim=len(dynamic_attributes_names) + len(training_data.list_static_attributes_names),
            image_height=training_data.max_dim, image_width=training_data.max_dim,
            hidden_dim=num_hidden_units, sequence_length_conv_lstm=sequence_length_spatial,
            in_cnn_channels=1, dropout=dropout,
            num_static_attributes=len(training_data.list_static_attributes_names),
            num_dynamic_attributes=len(dynamic_attributes_names))
    else:
        raise Exception(f"model with name {model_name} is not recognized")
    print(f"running with optimizer: {optim_name}")
    model = model.to(device="cuda")
    if optim_name.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # config = wandb.config
    # config.learning_rate = learning_rate
    # config.wandb = True
    # wandb.watch(model)
    train_dataloader = DataLoader(training_data, batch_size=256, num_workers=num_workers_data_loader, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=256, num_workers=num_workers_data_loader)
    if profile_code:
        p = profile(
            activities=[ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=2), on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler_logs"))
        p.start()
    best_median_nse = None
    list_training_loss_single_pass = []
    nse_list_last_pass = []
    for i in range(num_epochs):
        loss_on_training_epoch = train_epoch(model, optimizer, train_dataloader, calc_nse_star,
                                             epoch=(i + 1), device="cuda",
                                             print_tqdm_to_console=print_tqdm_to_console,
                                             specific_model_type=specific_model_type)
        if model_name.lower() == "cnn_lstm":
            print(f"the number of 'images' is: {model.cnn_lstm.number_of_images_counter}")
            model.cnn_lstm.number_of_images_counter = 0
        if profile_code:
            p.step()
        list_training_loss_single_pass.append(((i + 1), loss_on_training_epoch))
        if (i % calc_nse_interval) == (calc_nse_interval - 1):
            preds_obs_dict_per_basin = eval_model(model, test_dataloader, device="cuda", epoch=(i + 1),
                                                  print_tqdm_to_console=print_tqdm_to_console,
                                                  specific_model_type=specific_model_type)
            nse_list_last_pass, median_nse = calc_validation_basins_nse(preds_obs_dict_per_basin, (i + 1))
            if best_median_nse is None or best_median_nse < median_nse:
                best_median_nse = median_nse
            print(f"best NSE so far: {best_median_nse}")
        if profile_code:
            p.stop()
    return nse_list_last_pass, list_training_loss_single_pass


def seed_worker(worker_id):
    initialize_seed(123)


def choose_hyper_parameters_validation(
        static_attributes_names,
        dynamic_attributes_names,
        discharge_str,
        dynamic_data_folder_train,
        static_data_folder,
        discharge_data_folder,
        model_name,
        dataset_to_use,
        optim_name,
        shared_model,
        num_epochs,
        num_workers_data_loader,
        num_basins,
        profile_code,
        num_processes_ddp,
        create_new_files,
        sequence_length_spatial,
        print_tqdm_to_console,
        limit_size_above_1000,
        use_all_static_attr
):
    train_stations_list = []
    val_stations_list = []
    if dataset_to_use.lower() == "era5" or dataset_to_use.lower() == "caravan":
        all_stations_list_sorted = sorted(open("../data/531_basin_list.txt").read().splitlines())
    else:
        all_stations_list_sorted = sorted(open("../data/531_basin_list.txt").read().splitlines())
    all_stations_list_sorted = all_stations_list_sorted[:num_basins] if num_basins else all_stations_list_sorted
    # for i in range(len(all_stations_list_sorted)):
    #     if i % 5 != 0:
    #         train_stations_list.append(all_stations_list_sorted[i])
    #     else:
    #         val_stations_list.append(all_stations_list_sorted[i])
    train_stations_list = all_stations_list_sorted[:]
    val_stations_list = all_stations_list_sorted[:]
    learning_rates = np.linspace(5 * (10 ** -4), 5 * (10 ** -4), num=1).tolist()
    dropout_rates = [0.4]
    sequence_lengths = [270]
    if model_name.lower() == "transformer_lstm":
        num_hidden_units = [1]
    else:
        num_hidden_units = [256]
    dict_results = {
        "dropout rate": [],
        "sequence length": [],
        # "num epochs": [],
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
        )
    )
    curr_datetime = datetime.now()
    curr_datetime_str = curr_datetime.strftime("%d-%m-%Y_%H:%M:%S")
    for (
            learning_rate_param,
            dropout_rate_param,
            sequence_length_param,
            num_hidden_units_param,
    ) in all_parameters:
        nse_list_single_pass = run_single_parameters_check_with_val_on_years(
            train_stations_list,
            val_stations_list,
            sequence_length_param,
            learning_rate_param,
            num_hidden_units_param,
            dropout_rate_param,
            static_attributes_names,
            dynamic_attributes_names,
            discharge_str,
            dynamic_data_folder_train,
            static_data_folder,
            discharge_data_folder,
            num_epochs=num_epochs,
            model_name=model_name,
            dataset_to_use=dataset_to_use,
            optim_name=optim_name,
            shared_model=shared_model,
            num_workers_data_loader=num_workers_data_loader,
            profile_code=profile_code,
            num_processes_ddp=num_processes_ddp,
            create_new_files=create_new_files,
            sequence_length_spatial=sequence_length_spatial,
            print_tqdm_to_console=print_tqdm_to_console,
            limit_size_above_1000=limit_size_above_1000,
            use_all_static_attr=use_all_static_attr
        )
        if len(nse_list_single_pass) == 0:
            median_nse = -1
        else:
            median_nse = statistics.median(nse_list_single_pass)
        if len(nse_list_single_pass) > 0:
            list_plots_titles.append(
                f"{dropout_rate_param};"
                f"{sequence_length_param};"
                f"{num_hidden_units_param};"
                f"{num_epochs}"
            )
            list_nse_lists_basins.append(nse_list_single_pass)
        if median_nse > best_median_nse or best_median_nse == -1:
            best_median_nse = median_nse
            best_parameters = (
                dropout_rate_param,
                sequence_length_param,
                num_hidden_units_param,
                num_epochs,
            )
        dict_results["dropout rate"] = [dropout_rate_param]
        dict_results["sequence length"] = [sequence_length_param]
        dict_results["num hidden units"] = [num_hidden_units_param]
        # dict_results["num epochs"].append(num_epochs_param)
        dict_results["median NSE"] = [median_nse]
        plot_NSE_CDF(list_nse_lists_basins[-1], list_plots_titles[-1])
        plt.grid()
        plt.savefig(
            "../data/results/parameters_comparison" + f"_{list_plots_titles[-1]}" + ".png"
        )
        plt.close()
        df_results = pd.DataFrame(data=dict_results)
        df_results.to_csv(
            f"../data/results/results_{curr_datetime_str}.csv",
            mode="a",
            header=not os.path.exists(f"../data/results/results_{curr_datetime_str}.csv"),
        )
        print(f"best parameters: {best_parameters}")
    return best_parameters


def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
    print(output)
    p.export_chrome_trace("./trace_" + str(p.step_num) + ".json")


def initialize_seed(seed):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def main():
    # torch.backends.cudnn.enabled = False
    initialize_seed(123)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        help="which dataset to train and test / validate on",
        choices=["CAMELS", "CARAVAN"],
        default="ERA5",
    )
    parser.add_argument(
        "--model",
        help="which model to use",
        choices=["LSTM", "Transformer_LSTM", "CNN_LSTM", "CONV_LSTM", "Transformer_Seq2Seq"],
        default="LSTM",
    )
    parser.add_argument(
        "--optim",
        help="which optimizer to use",
        choices=["SGD", "Adam"],
        default="SGD",
    )
    parser.add_argument("--shared_model",
                        help="whether to run in shared model scenario - when the "
                             "training and validation stations are not the same",
                        choices=["True", "False"], default="False")
    parser.add_argument("--num_epochs", help="number of epochs for training", default=30, type=int)
    parser.add_argument("--num_workers_data_loader", help="number of workers for data loader", default=0, type=int)
    parser.add_argument("--num_basins", help="number of basins to include "
                                             "in training and test / validation", default=None, type=int)
    parser.add_argument('--profile_code', action='store_true')
    parser.add_argument("--num_processes_ddp", help="number of processes to run distributed data"
                                                    " parallelism", default=1, type=int)
    parser.add_argument("--sequence_length_spatial", help="the sequence length to take of spatial features",
                        default=7, type=int)
    parser.add_argument("--create_new_files", action="store_true")
    parser.add_argument("--print_tqdm_to_console", action="store_true")
    parser.add_argument("--limit_size_above_1000", action="store_true")
    parser.add_argument("--use_all_static_attr", action="store_true")
    parser.set_defaults(profile_code=False)
    parser.set_defaults(create_new_files=False)
    parser.set_defaults(print_tqdm_to_console=False)
    parser.set_defaults(limit_size_above_1000=False)
    parser.set_defaults(use_all_static_attr=False)
    command_args = parser.parse_args()
    if command_args.dataset == "CAMELS":
        choose_hyper_parameters_validation(
            CAMELS_dataset.STATIC_ATTRIBUTES_NAMES,
            CAMELS_dataset.DYNAMIC_ATTRIBUTES_NAMES,
            CAMELS_dataset.DISCHARGE_STR,
            CAMELS_dataset.DYNAMIC_DATA_FOLDER,
            CAMELS_dataset.STATIC_DATA_FOLDER,
            CAMELS_dataset.DISCHARGE_DATA_FOLDER,
            model_name=command_args.model,
            dataset_to_use="CAMELS",
            optim_name=command_args.optim,
            shared_model=bool(command_args.shared_model),
            num_epochs=command_args.num_epochs,
            num_workers_data_loader=command_args.num_workers_data_loader,
            num_basins=command_args.num_basins,
            profile_code=command_args.profile_code,
            num_processes_ddp=command_args.num_processes_ddp,
            create_new_files=command_args.create_new_files,
            sequence_length_spatial=command_args.sequence_length_spatial,
            print_tqdm_to_console=command_args.print_tqdm_to_console,
            limit_size_above_1000=command_args.limit_size_above_1000,
            use_all_static_attr=command_args.use_all_static_attr
        )
    elif command_args.dataset == "CARAVAN":
        choose_hyper_parameters_validation(
            ERA5_dataset.STATIC_ATTRIBUTES_NAMES,
            ERA5_dataset.DYNAMIC_ATTRIBUTES_NAMES_CARAVAN,
            ERA5_dataset.DISCHARGE_STR_CARAVAN,
            ERA5_dataset.DYNAMIC_DATA_FOLDER_CARAVAN,
            ERA5_dataset.STATIC_DATA_FOLDER,
            ERA5_dataset.DISCHARGE_DATA_FOLDER_CARAVAN,
            model_name=command_args.model,
            dataset_to_use="CARAVAN",
            optim_name=command_args.optim,
            shared_model=bool(command_args.shared_model),
            num_epochs=command_args.num_epochs,
            num_workers_data_loader=command_args.num_workers_data_loader,
            num_basins=command_args.num_basins,
            profile_code=command_args.profile_code,
            num_processes_ddp=command_args.num_processes_ddp,
            create_new_files=command_args.create_new_files,
            sequence_length_spatial=command_args.sequence_length_spatial,
            print_tqdm_to_console=command_args.print_tqdm_to_console,
            limit_size_above_1000=command_args.limit_size_above_1000,
            use_all_static_attr=command_args.use_all_static_attr
        )
    else:
        raise Exception(f"wrong dataset name: {command_args.dataset}")


if __name__ == "__main__":
    main()
