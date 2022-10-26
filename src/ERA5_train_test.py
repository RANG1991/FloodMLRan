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


def train_epoch(model, optimizer, loader, loss_func, epoch, device):
    # # set model to train mode (important for dropout)
    # model.train()
    pbar = tqdm_notebook(loader)
    print(f"Epoch {epoch}")
    pbar.set_description(f"Epoch {epoch}")
    loss_list = []
    # request mini-batch of data from the loader
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
        loss_list.append(loss.item())
        # write current loss in the progress bar
        print(f"Loss: {loss.item():.4f}")
        pbar.set_postfix_str(f"Loss: {loss.item():.4f}")
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
    df_all_data = read_basins_csv_files("../data/ERA5/all_data_daily", 3)
    df_all_data.to_csv("../data/df_train.csv")
    static_attributes_names = ["ele_mt_sav", "slp_dg_sav", "basin_area", "for_pc_sse",
                               "cly_pc_sav", "slt_pc_sav", "snd_pc_sav", "soc_th_sav",
                               "p_mean", "pet_mean",
                               "aridity", "frac_snow",
                               "high_prec_freq",
                               "high_prec_dur",
                               "low_prec_freq", "low_prec_dur"]
    training_data = Dataset_ERA5(dynamic_data_folder="../data/ERA5/all_data_daily",
                                 static_data_file_caravan="../data/ERA5/Caravan/attributes/attributes_caravan_us.csv",
                                 static_data_file_hydroatlas="../data/ERA5/Caravan/attributes"
                                                             "/attributes_hydroatlas_us.csv",
                                 static_attributes_names=static_attributes_names)
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LSTM_ERA5(hidden_dim=20, input_dim=len(static_attributes_names) + 1).to(device)
    learning_rate = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.MSELoss()
    loss_list = []
    for i in range(3):
        training_data.zero_out_accumulators()
        loss_list_epoch = train_epoch(model, optimizer, train_dataloader, loss_func, epoch=(i + 1), device=device)
        loss_list.extend(loss_list_epoch)
    plt.plot(loss_list)
    plt.show()


if __name__ == "__main__":
    main()
