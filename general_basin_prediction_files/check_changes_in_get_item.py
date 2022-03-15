from general_basin_prediction_files.Godavari import IMDGodavari as new_G
from general_basin_prediction_files.cnn_lstm_v3 import IMDGodavari as old_G
from pathlib import Path
import numpy as np
import torch
import tqdm.notebook
import tqdm
import os
from preprocess_data import Preprocessor
from matplotlib import pyplot as plt

root_dir = str(Path(os.getcwd()).parent)
RUN_LOCALLY = True
PATH_ROOT = root_dir + "/"  # Change only here the path
PATH_DATA_FILE = root_dir + str(Path("/" + "data/raw_data_fixed_17532_3_22_38"))
DIMS_JSON_FILE_PATH = root_dir + "./dims_json.json"
PATH_LABEL = PATH_ROOT + "data/CWC/"
PATH_LOC = PATH_ROOT + "data/LatLon/{0}_lat_lon"
PATH_DATA_CLEAN = PATH_ROOT + "data/IMD_Lat_Lon_reduced/"
PATH_MODEL = PATH_ROOT + "cnn_lstm/"
DISCHARGE_FORMAT = "CWC_discharge_{0}_clean"
PATH_CATCHMENTS = PATH_ROOT + "data/catchments.csv"
FILE_FORMAT = "data_{0}_{1}"
INCLUDE_STATIC = True

# Lat - width, Lon - height
LAT_MIN = 17.375
LAT_MAX = 22.625
LON_MIN = 73.625
LON_MAX = 82.875
GRID_DELTA = 0.25
LAT_GRID = np.arange(LAT_MIN, LAT_MAX + GRID_DELTA / 2, GRID_DELTA)
LON_GRID = np.arange(LON_MIN, LON_MAX + GRID_DELTA / 2, GRID_DELTA)
DATA_LEN = 17532
NUM_CHANNELS = 3
DEFAULT_LAT = len(LAT_GRID)
DEFAULT_LON = len(LON_GRID)


def train_epoch(device, model, optimizer, loader, loss_func, epoch):
    """Train model for a single epoch.

    :param device:
    :param model: A torch.nn.Module implementing the LSTM model
    :param optimizer: One of PyTorchs optimizer classes.
    :param loader: A PyTorch DataLoader, providing the trainings
        data in mini batches.
    :param loss_func: The loss function to minimize.
    :param epoch: The current epoch (int) used for the progress bar
    """
    # set model to train mode (important for dropout)
    model.train()
    pbar = tqdm.notebook.tqdm(loader)
    pbar.set_description(f"Epoch {epoch}")
    # request mini-batch of data from the loader
    for xs, ys in pbar:
        # delete previously stored gradients from the model
        optimizer.zero_grad()
        # push data to GPU (if available)
        xs, ys = xs.to(device), ys.to(device)
        # get model predictions
        y_hat = model(xs)
        # calculate loss
        loss = loss_func(y_hat, ys)
        # calculate gradients
        loss.backward()
        # update the weights
        optimizer.step()
        # write current loss in the progress bar
        pbar.set_postfix_str(f"Loss: {loss.item():.4f}")


def get_item_method_check():
    # Whether to use CPU or GPU. Use False for CPU mode.
    use_gpu = True
    device = torch.device("cuda:0" if (torch.cuda.is_available() and use_gpu) else "cpu")
    sequence_length = 30  # Length of the meteorological record provided to the network
    lead = 0  # 1
    # Choose features
    use_perc = True
    # maximum temperature in a given day
    use_t_max = False
    # minimum temperature in a given day
    use_t_min = False
    idx_features = [use_perc, use_t_max, use_t_min]
    basin_list = ['Tekra', 'Perur']
    start_date = (2000, 1, 1)
    end_date = (2001, 12, 31)
    preprocessor = Preprocessor(PATH_ROOT, idx_features, start_date, end_date, LAT_MIN,
                                LAT_MAX, LON_MIN, LON_MAX, GRID_DELTA, DATA_LEN, NUM_CHANNELS)
    # The data will always be in shape of - samples * channels * width * height
    all_data, image_width, image_height = \
        preprocessor.reshape_data_by_lat_lon_file(PATH_DATA_FILE, DIMS_JSON_FILE_PATH)
    catchment_dict = preprocessor.create_catchment_dict(PATH_CATCHMENTS)
    include_static = True
    # Training data
    months_lst = [6, 7, 8, 9, 10]
    # months_lst = []
    print('Train dataset\n===============================')
    ds_train_old = old_G(all_data,
                         basin_list=basin_list,
                         seq_length=sequence_length,
                         period="eval",
                         dates=[start_date, end_date],
                         months=months_lst,
                         idx=idx_features,
                         lead=lead,
                         include_static=INCLUDE_STATIC)
    ds_train_new = new_G(all_data,
                         catchment_dict=catchment_dict,
                         preprocessor=preprocessor,
                         basin_list=basin_list,
                         seq_length=sequence_length,
                         period="eval",
                         dates=[start_date, end_date],
                         months=months_lst,
                         idx=idx_features,
                         lead=lead,
                         include_static=include_static)
    print(str(ds_train_old.num_samples) + " " + str(ds_train_new.num_samples))
    for i in range(len(ds_train_new)):
        t1, y1 = ds_train_old[i]
        t2, y2 = ds_train_new[i]
        print("number of sample is: {}".format(i))
        t1_sum_reshaped = t1.sum(axis=0)[:-4].reshape(22, 38)
        t2_sum_reshaped = t2.sum(axis=0)[:-4].reshape(22, 38)
        abs_t1_t2 = np.abs(t1_sum_reshaped - t2_sum_reshaped)
        indices = np.argwhere(abs_t1_t2 > 0.01)
        if indices.numel() > 0:
            # plt.imshow(t1_sum_reshaped)
            # plt.show()
            # plt.imshow(t2_sum_reshaped)
            # plt.show()
            print("in sample number: {} in there is difference in X. The max difference is: {}".format(i,
                                        torch.max(abs_t1_t2)))
        abs_y1_y2 = np.abs(y1 - y2)
        y = np.argwhere(abs_y1_y2 > 0.01)
        if y.numel() > 0:
            print("in sample number: {} there is difference in Y".format(i))
        print("done with sample number: {}".format(i))


def main():
    get_item_method_check()


if __name__ == "__main__":
    main()
