import os

from torch.utils.data import DataLoader
from ERA5_dataset import Dataset_ERA5
from tqdm.notebook import tqdm as tqdm_notebook
from ERA5_lstm import LSTM_ERA5
import torch.optim
import torch.nn as nn
from os import listdir
from os.path import isfile, join
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
import numpy as np


def eval_model(model, loader, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the model.

    :param model: A torch.nn.Module implementing the LSTM model
    :param loader: A PyTorch DataLoader, providing the data.

    :return: Two torch Tensors, containing the observations and
        model predictions
    """
    # set model to eval mode (important for dropout)
    model.eval()
    obs = []
    preds = []
    # in inference mode, we don't need to store intermediate steps for
    # backprob
    with torch.no_grad():
        # request mini-batch of data from the loader
        for xs, ys in loader:
            # push data to GPU (if available)
            xs = xs.to(device)
            # get model predictions
            y_hat = model(xs)
            obs.append(ys)
            preds.append(y_hat)
    return torch.cat(obs), torch.cat(preds)


def calc_nse(obs: np.array, sim: np.array) -> float:
    """Calculate Nash-Sutcliff-Efficiency.

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


def train_epoch(model, optimizer, loader, loss_func, epoch, device):
    # # set model to train mode (important for dropout)
    model.train()
    pbar = tqdm_notebook(loader)
    print(f"Epoch {epoch}")
    pbar.set_description(f"Epoch {epoch}")
    loss_list = []
    # request mini-batch of data from the loader
    running_loss = 0.0
    i = 0
    for xs, ys in pbar:
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
            print(f'[{epoch + 1}] loss: {running_loss / 200:.3f}')
            loss_list.append(running_loss)
            running_loss = 0.0
        pbar.set_postfix_str(f"Loss: {loss.item():.4f}")
        i += 1
    return loss_list


def read_basins_csv_files(folder_name, num_basins):
    df = pd.DataFrame(columns=["date", "precip", "flow"])
    data_csv_files = [f for f in listdir(folder_name) if isfile(join(folder_name, f))]
    for i in range(min(num_basins, len(data_csv_files))):
        data_file = data_csv_files[i]
        if str(data_file).endswith(".csv"):
            df_temp = pd.read_csv(folder_name + os.sep + data_file)
            df = pd.concat([df, df_temp])
    return df


def main():
    # df_all_data = read_basins_csv_files("../data/ERA5/all_data_daily", 3)
    # df_all_data.to_csv("../data/df_train.csv")
    load_datasets_dynamically = False
    static_attributes_names = ["ele_mt_sav", "slp_dg_sav", "basin_area", "for_pc_sse",
                               "cly_pc_sav", "slt_pc_sav", "snd_pc_sav", "soc_th_sav",
                               "p_mean", "pet_mean",
                               "aridity", "frac_snow",
                               "high_prec_freq",
                               "high_prec_dur",
                               "low_prec_freq", "low_prec_dur"]
    training_data = Dataset_ERA5(dynamic_data_folder="../data/ERA5/all_data_daily/train/",
                                 static_data_file_caravan="../data/ERA5/Caravan/attributes/attributes_caravan_us.csv",
                                 static_data_file_hydroatlas="../data/ERA5/Caravan/attributes"
                                                             "/attributes_hydroatlas_us.csv",
                                 static_attributes_names=static_attributes_names,
                                 load_dynamically=load_datasets_dynamically, sequence_length=30)
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_data = Dataset_ERA5(dynamic_data_folder="../data/ERA5/all_data_daily/test/",
                             static_data_file_caravan="../data/ERA5/Caravan/attributes/attributes_caravan_us.csv",
                             static_data_file_hydroatlas="../data/ERA5/Caravan/attributes"
                                                         "/attributes_hydroatlas_us.csv",
                             static_attributes_names=static_attributes_names,
                             load_dynamically=load_datasets_dynamically,
                             x_maxs=training_data.get_x_max(), x_mins=training_data.get_x_min(),
                             y_mean=training_data.get_y_mean(), y_std=training_data.get_y_std(), sequence_length=30)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LSTM_ERA5(hidden_dim=20, input_dim=len(static_attributes_names) + 1).to(device)
    learning_rate = 5e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.MSELoss()
    loss_list = []
    for i in range(50):
        if load_datasets_dynamically:
            training_data.zero_out_accumulators()
            test_data.zero_out_accumulators()
        loss_list_epoch = train_epoch(model, optimizer, train_dataloader, loss_func, epoch=(i + 1), device=device)
        loss_list.extend(loss_list_epoch)
        obs, preds = eval_model(model, test_dataloader, device)
        nse = calc_nse(obs.cpu().numpy(), preds.cpu().numpy())
        print(f"NSE is: {nse}")
    plt.plot(loss_list)
    plt.show()


if __name__ == "__main__":
    main()
