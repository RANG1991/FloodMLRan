from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from os import listdir
from os.path import isfile, join
import re


class Dataset_ERA5(Dataset):

    def __init__(self, dynamic_data_folder, static_data_file_caravan,
                 static_data_file_hydroatlas, static_attributes_names=[],
                 is_training=True, sequence_length=270, load_dynamically=True,
                 x_mins=None, x_maxs=None, y_mean=None, y_std=None):
        self.is_training = is_training
        self.sequence_length = sequence_length
        self.dynamic_data_folder = dynamic_data_folder
        self.static_data_file_caravan = static_data_file_caravan
        self.static_data_file_hydroatlas = static_data_file_hydroatlas
        self.list_static_attributes_names = static_attributes_names
        self.df_attr, list_stations_static = self.read_static_attributes()
        all_station_files = [f for f in listdir(self.dynamic_data_folder) if isfile(join(dynamic_data_folder, f))]
        self.acum_stations_length = 0
        self.curr_station_index = 0
        self.X_data = np.array([])
        self.y_data = np.array([])
        self.load_dynamically = load_dynamically
        list_stations_dynamic = []
        X_data_list = []
        y_data_list = []
        for station_file_name in all_station_files:
            station_id = re.search("data24_(\\d+)\\.csv", station_file_name).group(1)
            if station_id in list_stations_static:
                list_stations_dynamic.append(station_id)
                if not self.load_dynamically:
                    X_data_curr, y_data_curr = self.read_single_station_file(station_id)
                    if X_data_curr.size > 0:
                        X_data_list.append(X_data_curr)
                        y_data_list.append(y_data_curr)
        if not self.load_dynamically:
            self.X_data = np.concatenate(X_data_list)
            self.y_data = np.concatenate(y_data_list)
            self.y_std = y_std if y_std is not None else self.y_data.std()
            self.y_mean = y_mean if y_mean is not None else self.y_data.mean()
            self.x_max = x_maxs if x_maxs is not None else self.X_data.max(axis=0)
            self.x_min = x_mins if x_mins is not None else self.X_data.min(axis=0)
            self.X_data = (self.X_data - self.x_min) / ((self.x_max - self.x_min) + (10**-10))
            self.y_data = (self.y_data - self.y_mean) / self.y_std
        self.list_stations = list(set(list_stations_static).intersection(set(list_stations_dynamic)))

    def __len__(self):
        return self.calculate_dataset_length(self.list_stations)

    def __getitem__(self, index) -> T_co:
        if self.load_dynamically:
            if (index - self.acum_stations_length) + self.sequence_length >= self.X_data.shape[0]:
                if self.curr_station_index > 0:
                    self.acum_stations_length += (self.X_data.shape[0] - self.sequence_length)
                self.X_data, self.y_data = self.read_single_station_file(self.list_stations[self.curr_station_index])
                self.y_std = self.y_data.std()
                self.y_mean = self.y_data.mean()
                self.x_max = self.X_data.max(axis=1)
                self.x_min = self.X_data.min(axis=1)
                self.curr_station_index += 1
            X_data_tensor = torch.tensor(self.X_data[(index - self.acum_stations_length):
                                                     (index - self.acum_stations_length) + self.sequence_length]).to(
                torch.float32)
            y_data_tensor = torch.tensor(self.y_data[(index - self.acum_stations_length) + self.sequence_length]).to(
                torch.float32)
        else:
            X_data_tensor = torch.tensor(self.X_data[index: index + self.sequence_length]).to(torch.float32)
            y_data_tensor = torch.tensor(self.y_data[index + self.sequence_length]).to(torch.float32)
        return X_data_tensor, y_data_tensor

    def read_static_attributes(self):
        df_attr_caravan = pd.read_csv(self.static_data_file_caravan, dtype={'gauge_id': str})
        df_attr_hydroatlas = pd.read_csv(self.static_data_file_hydroatlas, dtype={'gauge_id': str})
        df_attr = df_attr_caravan.merge(df_attr_hydroatlas, on="gauge_id")
        df_attr['gauge_id'] = df_attr['gauge_id'].apply(lambda x: str(x).replace("us_", "")).values.tolist()
        df_attr = df_attr.dropna()
        df_attr = df_attr[['gauge_id'] + self.list_static_attributes_names]
        maxes = df_attr.drop(columns=['gauge_id']).max(axis=1).to_numpy().reshape(-1, 1)
        mins = df_attr.drop(columns=['gauge_id']).min(axis=1).to_numpy().reshape(-1, 1)
        df_attr[self.list_static_attributes_names] = (df_attr.drop(columns=['gauge_id']).to_numpy() - mins) / (
                maxes - mins)
        return df_attr, df_attr['gauge_id'].values.tolist()

    def read_single_station_file(self, station_id):
        station_data_file = Path(self.dynamic_data_folder) / f"data24_{station_id}.csv"
        df_dynamic_data = pd.read_csv(station_data_file)
        print(df_dynamic_data.head())
        df_dynamic_data = df_dynamic_data.dropna()
        df_dynamic_data = df_dynamic_data.loc[df_dynamic_data["precip"] > 0]
        df_dynamic_data = df_dynamic_data.loc[df_dynamic_data["flow"] > 0]
        X_data = df_dynamic_data["precip"].to_numpy()
        y_data = df_dynamic_data["flow"].to_numpy().flatten()
        if X_data.size == 0 or y_data.size == 0:
            return np.array([]), np.array([])
        X_data = X_data.reshape(-1, 1)
        y_data = y_data.reshape(-1, 1)
        static_attrib_station = (self.df_attr[self.df_attr["gauge_id"] == station_id]).drop("gauge_id", axis=1) \
            .to_numpy().reshape(1, -1)
        static_attrib_station_rep = static_attrib_station.repeat(X_data.shape[0], axis=0)
        X_data = np.concatenate([X_data, static_attrib_station_rep], axis=1)
        return X_data, y_data

    def calculate_dataset_length(self, station_ids):
        count = 0
        for station_id in station_ids:
            station_data_file = Path(self.dynamic_data_folder) / f"data24_{station_id}.csv"
            df_dynamic_data = pd.read_csv(station_data_file)
            df_dynamic_data = df_dynamic_data.loc[df_dynamic_data["precip"] > 0]
            df_dynamic_data = df_dynamic_data.loc[df_dynamic_data["flow"] > 0]
            df_dynamic_data = df_dynamic_data.dropna()
            count += (len(df_dynamic_data.index) - self.sequence_length)
        return count

    def zero_out_accumulators(self):
        self.acum_stations_length = 0
        self.curr_station_index = 0
        self.X_data = np.array([])
        self.y_data = np.array([])

    def get_x_min(self):
        return self.x_min

    def get_x_max(self):
        return self.x_max

    def get_y_std(self):
        return self.y_std

    def get_y_mean(self):
        return self.y_mean
