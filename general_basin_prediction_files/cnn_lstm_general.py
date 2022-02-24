from pathlib import Path
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm.notebook
import tqdm
import datetime
import pathlib
import pytz
import os
import matplotlib.dates as mdates
from LSTM import CNNLSTM
from preprocess_data import Preprocessor
from Godavari import IMDGodavari
from captum.attr import IntegratedGradients
import seaborn as sns


root_dir = str(Path(os.getcwd()).parent)
RUN_LOCALLY = True
PATH_ROOT = root_dir + "/"  # Change only here the path
PATH_DATA_FILE = root_dir + str(Path("/" + "Data/raw_data_fixed_17532_3_22_38"))
DIMS_JSON_FILE_PATH = root_dir + "./dims_json.json"
PATH_LABEL = PATH_ROOT + "Data/CWC/"
PATH_LOC = PATH_ROOT + "Data/LatLon/{0}_lat_lon"
PATH_DATA_CLEAN = PATH_ROOT + "Data/IMD_Lat_Lon_reduced/"
PATH_MODEL = PATH_ROOT + "cnn_lstm/"
DISCHARGE_FORMAT = "CWC_discharge_{0}_clean"
PATH_CATCHMENTS = PATH_ROOT + "Data/catchments.csv"
FILE_FORMAT = "data_{0}_{1}"

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
DATA_START_DATE = (1967, 1, 1)
DATA_END_DATE = (2014, 12, 31)


def train_epoch(device, model, optimizer, loader, loss_func, epoch):
    """Train model for a single epoch.

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


def eval_model(device, model, loader) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the model.

    :param device:
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
    nse_val = 1 - numerator / denominator

    return nse_val


def calc_persist_nse(obs: np.array, sim: np.array, lead) -> float:
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

    # the reference is the last observed, instead of the mean
    sim = sim[lead:]
    obs = obs[lead:]
    ref = obs[:-lead]

    denominator = np.sum((obs - ref) ** 2)
    numerator = np.sum((sim - obs) ** 2)
    persist_nse_val = 1 - numerator / denominator

    return persist_nse_val


def calc_bias(obs: np.array, sim: np.array) -> float:
    """ Calculate bias

    :param obs: Array containing the observations
    :param sim: Array containing the simulations
    :return: NSE value.
    """
    bias_95 = None
    if np.percentile(obs, 95) != 0:
        bias_95 = (np.percentile(sim, 95) - np.percentile(obs, 95)) / np.percentile(obs, 95) * 100
    bias_5 = None
    if np.percentile(obs, 5) != 0:
        bias_5 = (np.percentile(sim, 5) - np.percentile(obs, 5)) / np.percentile(obs, 5) * 100
    mean_bias = None
    if np.nanmean(obs) != 0:
        mean_bias = (np.nanmean(sim) - np.nanmean(obs)) / np.nanmean(obs) * 100

    return bias_95, bias_5, mean_bias


def calc_maxdif(obs: np.array, sim: np.array) -> float:
    """ Calculate max difference in percent

    :param obs: Array containing the observations
    :param sim: Array containing the simulations
    :return: maxdif value.
    """
    max_sim = np.nanmax(sim)
    max_obs = np.nanmax(obs)

    return (max_sim - max_obs) / max_obs * 100


def calc_vol_qp(obs: np.array) -> float:
    """ Calculate volume [10^6 m^3] and peak discharge [m^3/s]

    :param obs: Array containing the observations
    :return: vol and qp values.
    """
    vol = np.nansum(obs) * 3600 * 24 / 1E6  # translate from m^3/s in daily resolution to 10^6 m^3
    qp = np.nanmax(obs)

    return vol, qp


def convert_to_number(number):
    try:
        ret_number = int(number)
        return ret_number
    except Exception:
        return None


def main():
    # Whether to use CPU or GPU. Use False for CPU mode.
    use_gpu = True
    device = torch.device("cuda:0" if (torch.cuda.is_available() and use_gpu) else "cpu")
    # Set random seed for reproducibility
    # manualSeed = 999
    # #manualSeed = random.randint(1, 10000) # use if you want new results
    # #print("Random Seed: ", manualSeed)
    # random.seed(manualSeed)
    # torch.manual_seed(manualSeed)
    #################################
    ###### Meta parameters ##########
    #################################
    hidden_size = 128  # Number of LSTM cells
    dropout_rate = 0.01  # Dropout rate of the final fully connected Layer [0.0, 1.0]
    # learning_rate = 2e-3 # Learning rate used to update the weights
    learning_rate = 1e-4  # Learning rate used to update the weights
    sequence_length = 30  # Length of the meteorological record provided to the network
    num_layers = 2  # Number of LSTM cells
    lead = 0  # 1
    cnn_outputsize = 20
    num_hidden_layers = 3
    num_hidden_units = 128
    ### Choose features ###
    use_perc = True
    # maximum temprature in a given day
    use_t_max = False
    # minimum temprature in a given day
    use_t_min = False
    idx_features = [use_perc, use_t_max, use_t_min]
    ### Choose basin ###
    # basin_list = ['Mancherial', 'Perur' ,'Pathagudem','Polavaram', 'Tekra']
    basin_list = ['Tekra', 'Perur']

    preprocessor = Preprocessor(PATH_ROOT, idx_features, DATA_START_DATE, DATA_END_DATE, LAT_MIN,
                                LAT_MAX, LON_MIN, LON_MAX, GRID_DELTA, DATA_LEN, NUM_CHANNELS)

    # The data will always be in shape of - samples * channels * width * height
    all_data, image_width, image_height = preprocessor.reshape_data_by_lat_lon_file(
        PATH_DATA_FILE, DIMS_JSON_FILE_PATH)

    ##############
    # Data set up#
    ##############
    catchment_dict = preprocessor.create_catchment_dict(PATH_CATCHMENTS)
    include_static = True

    # Training data
    start_date = (2000, 1, 1)
    end_date = (2009, 12, 31)
    months_lst = [6, 7, 8, 9, 10]

    print('Train dataset\n===============================')
    ds_train = IMDGodavari(all_data,
                           catchment_dict=catchment_dict,
                           preprocessor=preprocessor,
                           basin_list=basin_list,
                           seq_length=sequence_length,
                           period="train",
                           dates=[start_date, end_date],
                           months=months_lst,
                           idx=idx_features,
                           lead=lead,
                           include_static=include_static)
    tr_loader = DataLoader(ds_train, batch_size=64, shuffle=True)

    # Test data. We use the feature min/max of the training period for normalization
    start_date = (2010, 1, 1)
    end_date = (2014, 12, 31)
    print('\nTest dataset\n===============================')
    ds_test = IMDGodavari(all_data,
                          basin_list,
                          catchment_dict=catchment_dict,
                          preprocessor=preprocessor,
                          seq_length=sequence_length,
                          period="eval",
                          dates=[start_date, end_date],
                          months=months_lst,
                          idx=idx_features, lead=lead,
                          min_values=ds_train.get_min(),
                          max_values=ds_train.get_max(),
                          mean_y=ds_train.get_mean_y(),
                          std_y=ds_train.get_std_y(),
                          include_static=include_static)
    test_loader = DataLoader(ds_test, batch_size=2048, shuffle=False)

    #########################
    # Model, Optimizer, Loss#
    #########################

    # Here we create our model
    # attributes == static features
    num_attributes = catchment_dict['Tekra'].size

    if not include_static:
        num_attributes = 0

    # idx_features - a True / False list over the 3 features (channels) of each "image"
    input_size = (sum(idx_features) * DEFAULT_LON * DEFAULT_LAT + num_attributes) * sequence_length
    input_image_size = (sum(idx_features), image_width, image_height)
    model = CNNLSTM(lat=image_width, lon=image_height, input_size=cnn_outputsize, num_layers=num_layers,
                    hidden_size=hidden_size,
                    dropout_rate=dropout_rate, num_channels=sum(idx_features),
                    num_attributes=num_attributes, image_input_size=input_image_size).to(device)
    # model = DNN(input_size=input_size, num_hidden_layers=num_hidden_layers,
    # num_hidden_units=num_hidden_units,
    # dropout_rate=dropout_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.MSELoss()

    n_epochs = 50  # Number of training epochs

    # Creating the checkpoint folders
    datetime_israel = datetime.datetime.now(pytz.timezone('Israel'))
    path_train_ckpt = PATH_MODEL + datetime_israel.strftime("%Y_%m_%d-%H-%M-%S/")
    pathlib.Path(path_train_ckpt).mkdir(parents=True, exist_ok=True)

    for i in range(n_epochs):
        train_epoch(device, model, optimizer, tr_loader, loss_func, i + 1)
        obs, preds = eval_model(device, model, test_loader)
        preds = ds_test.revert_y_normalization(preds.cpu().numpy(), "")
        nse = calc_nse(obs.numpy(), preds)
        tqdm.tqdm.write(f"Test NSE: {nse:.3f}")
        model_name = "epoch_{:d}_nse_{:.3f}.ckpt".format(i + 1, nse)
        torch.save(model, path_train_ckpt + model_name)
        last_model_path = path_train_ckpt + model_name

    # start_date = (2000, 1, 1)
    # end_date = (2014, 12, 31)
    start_date = (2000, 1, 1)
    end_date = (2009, 12, 31)
    months_lst = [6, 7, 8, 9, 10]

    Validation_basin = ["Tekra"]
    ds_val = IMDGodavari(all_data,
                         basin_list=Validation_basin,
                         catchment_dict=catchment_dict,
                         preprocessor=preprocessor,
                         seq_length=sequence_length,
                         period="eval",
                         dates=[start_date, end_date],
                         months=months_lst,
                         idx=idx_features,
                         lead=lead,
                         min_values=ds_train.get_min(),
                         max_values=ds_train.get_max(),
                         mean_y=ds_train.get_mean_y(),
                         std_y=ds_train.get_std_y(),
                         include_static=include_static)
    val_loader = DataLoader(ds_val, batch_size=2048, shuffle=False)
    obs, preds = eval_model(device, model, val_loader)
    preds = ds_val.revert_y_normalization(preds.cpu().numpy(), "")
    obs = obs.numpy()
    nse = calc_nse(obs, preds)
    pb95, pb5, total_b = calc_bias(obs, preds)
    # Plot results
    start_date_tpl = ds_val.dates[0]
    start_date = pd.to_datetime(
        datetime.datetime(start_date_tpl[0], start_date_tpl[1], start_date_tpl[2], 0, 0)) + pd.DateOffset(
    )
    end_date_tpl = ds_val.dates[1]
    temp = pd.to_datetime(datetime.datetime(end_date_tpl[0], end_date_tpl[1], end_date_tpl[2], 0, 0))
    end_date = temp + pd.DateOffset()
    date_range = pd.date_range(start_date, end_date)
    # months = get_months_by_dates(start_date, end_date)
    ind_include = [i for i in range(0, len(date_range)) if date_range[i].month in months_lst]
    date_range = date_range[ind_include]
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.plot(date_range, obs, label="observation")
    ax.plot(date_range, preds, label="prediction")
    ax.legend()
    ax.set_title(
        f"Basin {Validation_basin} - Validation set NSE: {nse:.3f}, "
        f"95bias: {pb95:.1f}, 5bias: {pb5:.3f} ,total bias : {total_b: .1f}%")
    ax.xaxis.set_tick_params(rotation=90)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.set_xlabel("Date")
    ax.grid('on')
    _ = ax.set_ylabel("Discharge (mm/d)")
    """# Integrated gradients"""
    # Calculate Integrated Gradients
    start_date_ig = (2012, 8, 26)
    end_date_ig = (2012, 9, 5)
    model.eval()
    model.cpu()
    ig = IntegratedGradients(model, multiply_by_inputs=True)
    basline = torch.zeros(ds_val.x[idx[0]:idx[0] + 1, :, :].shape)
    integ_grad = np.zeros(ds_val.x[idx[0]:idx[0] + 1, :, :].shape)
    for i in idx:
        integ_grad += ig.attribute(ds_val.x[i:(i + 1), :, :], basline).numpy()
    integ_grad = np.squeeze(integ_grad)
    integ_grad /= len(idx)
    _ = model.cuda()

    image_grad = integ_grad[:, :DEFAULT_LAT * DEFAULT_LON].reshape((sequence_length, DEFAULT_LAT, DEFAULT_LON))
    time_vector_grad = np.sum(image_grad.reshape((image_grad.shape[0], image_grad.shape[1] * image_grad.shape[2])), axis=1)
    spatial_image_grad = np.sum(image_grad, axis=0)
    atrrib_grade = integ_grad[:, DEFAULT_LAT * DEFAULT_LON:]

    predsmonsoon = preds[np.where((date_range.month >= 6) & (date_range.month <= 10))[0]]
    obsmonsoon = obs[np.where((date_range.month >= 6) & (date_range.month <= 10))[0]]
    threshq1 = np.percentile(predsmonsoon, 90)
    threshq2 = np.percentile(predsmonsoon, 55)
    # idx = np.asarray([i for i in range(0,len(preds)) if (preds[i]>threshq1) & (preds[i]<threshq2)])
    idx = np.asarray([i for i in range(0, len(preds)) if (preds[i] > threshq1)])
    # idx = np.where((preds>threshq1) & (preds<threshq2) & (date_range.month>=6) & (date_range.month<=10))[0]
    print([threshq1, threshq2, idx.shape])
    # set model to eval mode (important for dropout)
    model.eval()
    model.cpu()
    ig = IntegratedGradients(model, multiply_by_inputs=True)
    basline = torch.zeros(ds_val.x[idx[0]:idx[0] + 1, :, :].shape)
    integ_grad = np.zeros(ds_val.x[idx[0]:idx[0] + 1, :, :].shape)
    for i in idx:
        # print (i)
        integ_grad += ig.attribute(ds_val.x[i:(i + 1), :, :], basline).numpy()
    integ_grad = np.squeeze(integ_grad)
    integ_grad /= len(idx)
    _ = model.cuda()

    image_grad = integ_grad[:, :DEFAULT_LAT * DEFAULT_LON].reshape((sequence_length, DEFAULT_LAT, DEFAULT_LON))
    time_vector_grad = np.sum(image_grad.reshape((image_grad.shape[0], image_grad.shape[1] * image_grad.shape[2])), axis=1)
    spatial_image_grad = np.sum(image_grad, axis=0)
    atrrib_grade = integ_grad[:, DEFAULT_LAT * DEFAULT_LON:]

    # integ_file = PATH_ROOT + "Out/integ_grad_2000_2014"
    # np.save(file=integ_file, arr=integ_grad)

    # Plot Integrated Gradients - Spatial
    sequence_length_small = 9
    image_grad_small = image_grad[sequence_length - sequence_length_small:, :]
    n_w_win = 3
    n_h_win = int((sequence_length_small + 1) / n_w_win)
    fig, ax = plt.subplots(n_h_win, n_w_win, figsize=(10 * n_h_win, 6 * n_w_win))
    max_v = abs(image_grad_small).max()
    min_v = -max_v
    for i in range(sequence_length_small):
        ax.flat[i].set_title(f'Day {i - sequence_length_small}')
        df = pd.DataFrame(image_grad_small[i, :], index=list(LAT_GRID), columns=list(LON_GRID))
        sns.heatmap(df[::-1], ax=ax.flat[i], vmin=min_v, vmax=max_v, square=True, cmap='RdYlBu')

    # plot without catchment attributes
    with plt.style.context('ggplot'):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))
        # ax1.plot(new_date_range, obs[idx], label="observation")
        # ax1.plot(new_date_range, preds[idx], label="prediction")
        # ax1.plot(idx, obs[idx], 'x', label="observation")
        # ax1.plot(idx, preds[idx],'x', label="prediction")
        ax1.plot(idx, obs[idx], 'x', label="observation")
        ax1.plot(idx, preds[idx], 'x', label="prediction")
        ax1.legend()
        #  ax1.set_title(f"Basin {Validation_basin} discharge>q95")
        ax1.set_title(f"Basin {Validation_basin}, monsoon, discharge: >q90")
        # ax1.xaxis.set_tick_params(rotation=90)
        ax1.set_xlabel("Date")
        ax1.grid('on')
        _ = ax1.set_ylabel("Discharge (mm/d)")

        #  df = pd.DataFrame(spatial_image_grad, index =list(LAT_GRID), columns =list(LON_GRID))
        df = pd.DataFrame(image_grad_small[sequence_length_small - 3, :], index=list(LAT_GRID), columns=list(LON_GRID))
        #  vmax=np.max(np.abs(spatial_image_grad.flat))
        vmax = np.max(np.abs(image_grad_small[sequence_length_small - 3, :].flat))
        sns.heatmap(df[::-1], ax=ax2, square=True, cmap='RdYlBu', vmin=-vmax, vmax=vmax)

        txn = np.arange(-sequence_length + 1, 0 + 1)
        ax3.plot(txn, time_vector_grad, '-o', color='c')
        ax3.set_xlabel("Day")
        ax3.set_ylabel("Integrated gradients")
        ax3.set_xlim([-sequence_length + 1, 0])
        ax3.set_xticks(txn)

    # plot
    with plt.style.context('ggplot'):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))
        ax1.plot(new_date_range, obs[idx], label="observation")
        ax1.plot(new_date_range, preds[idx], label="prediction")
        ax1.legend()
        ax1.set_title(f"Basin {Validation_basin}")
        # ax1.xaxis.set_tick_params(rotation=90)
        ax1.set_xlabel("Date")
        ax1.grid('on')
        _ = ax1.set_ylabel("Discharge (mm/d)")

        ax2.bar(['Precipitation', "Mean Precipitation", "Aridity", "Area", "Mean elevation"], sum(att),
                color=['b', 'g', 'g', 'g', 'g'])
        ax2.plot(['Precipitation', "Mean Precipitation", "Aridity", "Area", "Mean elevation"], [0, 0, 0, 0, 0], 'k')
        ax2.set_ylabel("Attribute sum integrated gradients")

        txn = np.arange(-sequence_length + 1, 0 + 1)
        ax3.plot(txn, att[:, 0], '-o', color='c')
        ax3.set_xlabel("Day")
        ax3.set_ylabel("Integrated gradients")
        ax3.set_xlim([-sequence_length + 1, 0])
        ax3.set_xticks(txn)

    # This cell is for creating the raw data - no need to run this
    start_date_pd = pd.to_datetime(datetime.datetime(DATA_START_DATE[0], DATA_START_DATE[1], DATA_START_DATE[2], 0, 0))
    end_date_pd = pd.to_datetime(datetime.datetime(DATA_END_DATE[0], DATA_END_DATE[1], DATA_END_DATE[2], 0, 0))
    date_range = pd.date_range(start_date_pd, end_date_pd)
    num_days = len(date_range)
    num_features = 3
    h = len(LAT_GRID)
    w = len(LON_GRID)
    data = np.zeros((num_days, num_features, h, w))
    for i, lat_i in enumerate(LAT_GRID):
        for j, lon_j in enumerate(LON_GRID):
            x = get_geo_raw_data(lat_i, lon_j, start_date, end_date)
            data[:, :, i, j] = x
    out_path = PATH_ROOT + 'Data/'
    data.tofile(out_path + "raw_data_fixed" + '_'.join([str(_) for _ in data.shape]))


if __name__ == '__main__':
    main()

# """# Integrated gradients"""
# # Calculate Integrated Gradients
#
# start_date_ig = (2012, 8, 26)
# end_date_ig = (2012, 9, 5)
# model.eval()
# model.cpu()
# ig = IntegratedGradients(model, multiply_by_inputs=True)
# basline = torch.zeros(ds_val.x[idx[0]:idx[0] + 1, :, :].shape)
# integ_grad = np.zeros(ds_val.x[idx[0]:idx[0] + 1, :, :].shape)
# for i in idx:
#     integ_grad += ig.attribute(ds_val.x[i:(i + 1), :, :], basline).numpy()
# integ_grad = np.squeeze(integ_grad)
# integ_grad /= len(idx)
# _ = model.cuda()
#
# image_grad = integ_grad[:, :DEFAULT_LAT * DEFAULT_LON].reshape((sequence_length, DEFAULT_LAT, DEFAULT_LON))
# time_vector_grad = np.sum(image_grad.reshape((image_grad.shape[0], image_grad.shape[1] * image_grad.shape[2])), axis=1)
# spatial_image_grad = np.sum(image_grad, axis=0)
# atrrib_grade = integ_grad[:, DEFAULT_LAT * DEFAULT_LON:]
#
# predsmonsoon = preds[np.where((date_range.month >= 6) & (date_range.month <= 10))[0]]
# obsmonsoon = obs[np.where((date_range.month >= 6) & (date_range.month <= 10))[0]]
# threshq1 = np.percentile(predsmonsoon, 90)
# threshq2 = np.percentile(predsmonsoon, 55)
# # idx = np.asarray([i for i in range(0,len(preds)) if (preds[i]>threshq1) & (preds[i]<threshq2)])
# idx = np.asarray([i for i in range(0, len(preds)) if (preds[i] > threshq1)])
# # idx = np.where((preds>threshq1) & (preds<threshq2) & (date_range.month>=6) & (date_range.month<=10))[0]
# print([threshq1, threshq2, idx.shape])
# # set model to eval mode (important for dropout)
# model.eval()
# model.cpu()
# ig = IntegratedGradients(model, multiply_by_inputs=True)
# basline = torch.zeros(ds_val.x[idx[0]:idx[0] + 1, :, :].shape)
# integ_grad = np.zeros(ds_val.x[idx[0]:idx[0] + 1, :, :].shape)
# for i in idx:
#     # print (i)
#     integ_grad += ig.attribute(ds_val.x[i:(i + 1), :, :], basline).numpy()
# integ_grad = np.squeeze(integ_grad)
# integ_grad /= len(idx)
# _ = model.cuda()
#
# image_grad = integ_grad[:, :DEFAULT_LAT * DEFAULT_LON].reshape((sequence_length, DEFAULT_LAT, DEFAULT_LON))
# time_vector_grad = np.sum(image_grad.reshape((image_grad.shape[0], image_grad.shape[1] * image_grad.shape[2])), axis=1)
# spatial_image_grad = np.sum(image_grad, axis=0)
# atrrib_grade = integ_grad[:, DEFAULT_LAT * DEFAULT_LON:]
#
# # integ_file = PATH_ROOT + "Out/integ_grad_2000_2014"
# # np.save(file=integ_file, arr=integ_grad)
#
# # Plot Integrated Gradients - Spatial
# sequence_length_small = 9
# image_grad_small = image_grad[sequence_length - sequence_length_small:, :]
# n_w_win = 3
# n_h_win = int((sequence_length_small + 1) / n_w_win)
# fig, ax = plt.subplots(n_h_win, n_w_win, figsize=(10 * n_h_win, 6 * n_w_win))
# max_v = abs(image_grad_small).max()
# min_v = -max_v
# for i in range(sequence_length_small):
#     ax.flat[i].set_title(f'Day {i - sequence_length_small}')
#     df = pd.DataFrame(image_grad_small[i, :], index=list(LAT_GRID), columns=list(LON_GRID))
#     sns.heatmap(df[::-1], ax=ax.flat[i], vmin=min_v, vmax=max_v, square=True, cmap='RdYlBu')
#
# # plot without catchment attributes
# with plt.style.context('ggplot'):
#     fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))
#     # ax1.plot(new_date_range, obs[idx], label="observation")
#     # ax1.plot(new_date_range, preds[idx], label="prediction")
#     # ax1.plot(idx, obs[idx], 'x', label="observation")
#     # ax1.plot(idx, preds[idx],'x', label="prediction")
#     ax1.plot(idx, obs[idx], 'x', label="observation")
#     ax1.plot(idx, preds[idx], 'x', label="prediction")
#     ax1.legend()
#     #  ax1.set_title(f"Basin {Validation_basin} discharge>q95")
#     ax1.set_title(f"Basin {Validation_basin}, monsoon, discharge: >q90")
#     # ax1.xaxis.set_tick_params(rotation=90)
#     ax1.set_xlabel("Date")
#     ax1.grid('on')
#     _ = ax1.set_ylabel("Discharge (mm/d)")
#
#     #  df = pd.DataFrame(spatial_image_grad, index =list(LAT_GRID), columns =list(LON_GRID))
#     df = pd.DataFrame(image_grad_small[sequence_length_small - 3, :], index=list(LAT_GRID), columns=list(LON_GRID))
#     #  vmax=np.max(np.abs(spatial_image_grad.flat))
#     vmax = np.max(np.abs(image_grad_small[sequence_length_small - 3, :].flat))
#     sns.heatmap(df[::-1], ax=ax2, square=True, cmap='RdYlBu', vmin=-vmax, vmax=vmax)
#
#     txn = np.arange(-sequence_length + 1, 0 + 1)
#     ax3.plot(txn, time_vector_grad, '-o', color='c')
#     ax3.set_xlabel("Day")
#     ax3.set_ylabel("Integrated gradients")
#     ax3.set_xlim([-sequence_length + 1, 0])
#     ax3.set_xticks(txn)
#
# # plot
# with plt.style.context('ggplot'):
#     fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))
#     ax1.plot(new_date_range, obs[idx], label="observation")
#     ax1.plot(new_date_range, preds[idx], label="prediction")
#     ax1.legend()
#     ax1.set_title(f"Basin {Validation_basin}")
#     # ax1.xaxis.set_tick_params(rotation=90)
#     ax1.set_xlabel("Date")
#     ax1.grid('on')
#     _ = ax1.set_ylabel("Discharge (mm/d)")
#
#     ax2.bar(['Precipitation', "Mean Precipitation", "Aridity", "Area", "Mean elevation"], sum(att),
#             color=['b', 'g', 'g', 'g', 'g'])
#     ax2.plot(['Precipitation', "Mean Precipitation", "Aridity", "Area", "Mean elevation"], [0, 0, 0, 0, 0], 'k')
#     ax2.set_ylabel("Attribute sum integrated gradients")
#
#     txn = np.arange(-sequence_length + 1, 0 + 1)
#     ax3.plot(txn, att[:, 0], '-o', color='c')
#     ax3.set_xlabel("Day")
#     ax3.set_ylabel("Integrated gradients")
#     ax3.set_xlim([-sequence_length + 1, 0])
#     ax3.set_xticks(txn)
#
# # This cell is for creating the raw data - no need to run this
# start_date_pd = pd.to_datetime(datetime.datetime(DATA_START_DATE[0], DATA_START_DATE[1], DATA_START_DATE[2], 0, 0))
# end_date_pd = pd.to_datetime(datetime.datetime(DATA_END_DATE[0], DATA_END_DATE[1], DATA_END_DATE[2], 0, 0))
# date_range = pd.date_range(start_date_pd, end_date_pd)
# num_days = len(date_range)
# num_features = 3
# h = len(LAT_GRID)
# w = len(LON_GRID)
# data = np.zeros((num_days, num_features, h, w))
# for i, lat_i in enumerate(LAT_GRID):
#     for j, lon_j in enumerate(LON_GRID):
#         x = get_geo_raw_data(lat_i, lon_j, start_date, end_date)
#         data[:, :, i, j] = x
# out_path = PATH_ROOT + 'Data/'
# data.tofile(out_path + "raw_data_fixed" + '_'.join([str(_) for _ in data.shape]))
