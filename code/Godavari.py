from torch.utils.data import Dataset
import copy
from typing import List, Dict
import numpy as np
import torch
from preprocess_data import Preprocessor
import pandas as pd
import datetime


class IMDGodavari(Dataset):
    """
    Torch Dataset for basic use of data from the data set.
    This data set provides meteorological observations and discharge
    of a given basin from the IMD Godavari data set.
    """

    def __init__(self, all_data: np.array,
                 basin_list: List,
                 preprocessor: Preprocessor,
                 catchment_dict: Dict,
                 seq_length: int,
                 period: str = None,
                 dates: List = None,
                 months: List = None,
                 idx: list = [True, True, True],
                 lead=0,
                 mask_list=[0, 0.5, 0.5],
                 include_static: np.bool = True):
        """Initialize Dataset containing the data of a single basin.
        :param basin_list: List of basins.
        :param seq_length: Length of the time window of meteorological
        input provided for one time step of prediction.
        (currently it's 30 - 30 days)
        :param period: (optional) One of ['train', 'eval']. None loads the entire time series.
        :param dates: (optional) List of the start and end date of the discharge period that is used.
        """
        self.max_values = None
        self.min_values = None
        self.num_samples = 0
        self.x = None
        self.y = None
        self.num_attributes = 0
        self.basin_list = basin_list
        self.preprocessor = preprocessor
        self.catchment_dict = catchment_dict
        self.seq_length = seq_length
        self.period = period
        self.dates = dates
        self.months = months
        self.idx_features = idx
        self.lead = lead
        self.mask_list = mask_list
        self.num_features = None
        self.include_static = include_static
        self.sample_to_basin_x = {}
        self.sample_to_basin_name_y = {}
        self.basin_name_to_mean_std_y = {}
        self.load_data(all_data)

    # The number of samples is: (the original number of samples
    # (timestamps) - sequence length) * number of basins
    # each samples is of size: (sequence length * number of features
    # ((width * height * channels) + static features))
    def __len__(self):
        return self.num_samples

    # each call to __getitem__ should return ONE sample, which is of size
    # (sequence length * number of features (channels * width * height))
    def __getitem__(self, idx: int):
        x_sample = None
        y_label = None
        for key in self.sample_to_basin_x.keys():
            start_ind = key[0]
            end_ind = key[1]
            if idx in range(start_ind, end_ind):
                indices_X, static_features, _ = self.sample_to_basin_x[key]
                basin_name, y = self.sample_to_basin_name_y[key]
                idx = idx - start_ind
                x_sample = copy.deepcopy(self.x[idx: idx + self.seq_length, :, :, :])
                y_label = y[idx]
                x_sample = np.multiply(x_sample, indices_X)
                x_sample = np.reshape(x_sample,
                                      (x_sample.shape[0], x_sample.shape[1], x_sample.shape[2] * x_sample.shape[3]))
                if self.period == 'train':
                    for i in range(x_sample.shape[1]):
                        x_sample[:, i, :] -= self.min_values[i]
                        x_sample[:, i, :] /= (self.max_values[i] - self.min_values[i])
                x_sample = np.reshape(x_sample, (x_sample.shape[0], x_sample.shape[1] * x_sample.shape[2]))
                if self.include_static:
                    static_features = static_features[np.newaxis, :]
                    static_features = np.repeat(static_features, x_sample.shape[0], axis=0)
                    x_sample = np.concatenate([x_sample, static_features], axis=1)
                break
        end_indices = [key[1] for key in self.sample_to_basin_x.keys()]
        start_indices = [key[0] for key in self.sample_to_basin_x.keys()]
        if x_sample is None or y_label is None:
            print("Error - the requested to be retrieved is not in the "
                  "range of start_ind, end_ind. "
                  "The requested index is: {}, the start indices are: "
                  "{}, the end indices are: {}".format(idx,
                                                       start_indices,
                                                       end_indices))
        x_sample = torch.from_numpy(x_sample.astype(np.float32))
        y_label = torch.tensor(y_label.astype(np.float32))
        return x_sample, y_label

    def load_data(self, all_data):
        """Load input and output data from text files"""
        start_date, end_date = self.dates
        idx_s = self.preprocessor.get_index_by_date(start_date)[0]
        idx_e = self.preprocessor.get_index_by_date(end_date)[0]
        # cropping the data to only the interested dates and the interested idx_features,
        # the idx_features are the "channels" (3 features - minimum precipitation,
        # temperature, maximum temperature)
        data = all_data[idx_s:idx_e + 1, self.idx_features, :, :]
        self.x = copy.deepcopy(all_data[idx_s:idx_e + 1, self.idx_features, :, :])
        self.num_features = data.shape[1] * data.shape[2] * data.shape[3]
        indices_to_include_months = []
        for i, basin in enumerate(self.basin_list):
            indices_X, static_features = self.get_basin_indices_x_and_static_features(basin)
            indices_X_time_features = np.ones((self.seq_length, len([x for x in self.idx_features if x]),
                                               *indices_X.shape))
            indices_X_time_features[:, :, :, :] = indices_X
            # calculating the min / max over all the channels - i.e. -
            # over all the samples (timestamps) + H_LAT + W_LON per channel
            self.min_values = self.x.min(axis=0).min(axis=1).min(axis=1)
            self.max_values = self.x.max(axis=0).max(axis=1).max(axis=1)
            y = self.preprocessor.get_basin_indices_y(basin, start_date, end_date)
            if self.period == 'train':
                mu_y = y.mean()
                print("Y mean is:", mu_y)
                std_y = y.std()
                print("Y std is:", std_y)
                y = ((y - mu_y) / std_y)
                self.basin_name_to_mean_std_y[basin] = (mu_y, std_y)
            indices_to_include_months, y_new = self.extract_from_data_by_months(y, start_date, end_date,
                                                                                basin_name=basin, basin_number=i)
            self.sample_to_basin_x[(i * (len(indices_to_include_months)),
                                    (i + 1) * (len(indices_to_include_months)))] = \
                (indices_X_time_features, static_features, indices_to_include_months)
            self.sample_to_basin_name_y[(i * (len(indices_to_include_months)),
                                         (i + 1) * (len(indices_to_include_months)))] = \
                (basin, y_new)

        self.num_samples = len(self.basin_list) * (len(indices_to_include_months))
        if not self.include_static:
            self.num_attributes = 0
        indices_to_include_months = range(indices_to_include_months[0],
                                          indices_to_include_months[-1] + self.seq_length + self.lead)
        self.x = self.x[indices_to_include_months, :, :, :]
        self.time_span = self.x.shape[0]
        print("data set for {0} for basins: {1}".format(self.period, self.basin_list))
        print("Number of attributes should be: {0}".format(self.num_attributes))
        print("Number of features should be: num_features + num_attributes= {0}".format(
            self.num_features + self.num_attributes))
        print("Number of sample should be: (time_span - sequence_len + 1 -lead) x num_basins= {0}".format(
            (self.time_span - self.seq_length + 1 - self.lead) * len(self.basin_list)))

    def get_basin_indices_x_and_static_features(self, basin):
        indices_X = self.preprocessor.get_basin_indices_x(basin)
        x_static = (self.catchment_dict[basin] - self.catchment_dict['mean']) / self.catchment_dict['std']
        self.num_attributes = len(x_static)
        if not self.include_static:
            self.num_attributes = 0
            x_static = None
        return indices_X, x_static

    def revert_y_normalization(self, y_orig: np.ndarray, basin_name="") -> np.ndarray:
        if basin_name not in self.basin_name_to_mean_std_y.keys():
            raise RuntimeError(
                f"Unknown Basin {basin_name}, the "
                f"training data was trained on {list(self.basin_name_to_mean_std_y.keys())}")
        y_mean = self.basin_name_to_mean_std_y[basin_name][0]
        y_std = self.basin_name_to_mean_std_y[basin_name][1]
        print("Y mean and std in renormalization is: {} {}".format(y_mean, y_std))
        y = (y_orig * y_std) + y_mean
        return y

    def extract_from_data_by_months(self, y, start_date, end_date, basin_name, basin_number):
        # reverting normalization to Y as we are going to normalize it again at the end
        date_months = self.get_months_by_dates(start_date, end_date)
        # adjusting for sequence length and lead
        date_months = date_months[(self.seq_length + self.lead - 1):]
        n_samples_per_basin = len(y) - (self.seq_length + self.lead - 1)
        ind_date_months = [i for i in range(0, n_samples_per_basin)]
        y_new = np.zeros((n_samples_per_basin,))
        y_new[:] = y[(self.seq_length + self.lead - 1):]
        np.savetxt("./y_labels_before_renormalization_in_new_ds", y_new, delimiter=",")
        if len(self.months) == 0:
            return ind_date_months, y_new
        if self.period == 'train':
            y_new = self.revert_y_normalization(y_new, basin_name)
        # getting the months for each date
        idx_temp = [idx for idx in ind_date_months if date_months[idx] in self.months]
        y_new = y_new[idx_temp]
        if self.period == 'train':
            mu_y_temp = y_new.mean()
            std_y_temp = y_new.std()
            y_new = (y_new - mu_y_temp) / std_y_temp
            self.basin_name_to_mean_std_y[basin_name] = (mu_y_temp, std_y_temp)
        return idx_temp, y_new

    @staticmethod
    def get_months_by_dates(start_date, end_date):
        start_date_pd = pd.to_datetime(datetime.datetime(start_date[0], start_date[1], start_date[2], 0, 0))
        end_date_pd = pd.to_datetime(datetime.datetime(end_date[0], end_date[1], end_date[2], 0, 0))
        date_range = pd.date_range(start_date_pd, end_date_pd)
        months = [date_range[i].month for i in range(0, len(date_range))]
        return months

    def get_num_features(self):
        return self.num_features
