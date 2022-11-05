from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from os import listdir
from os.path import isfile, join
import re
from datetime import datetime

class Dataset_ERA5(Dataset):

    def __init__(self, dynamic_data_folder, static_data_file_caravan,
                 static_data_file_hydroatlas,
                 dynamic_attributes_names, discharge_str,
                 train_start_date,
                 train_end_date,
                 validation_start_date,
                 validation_end_date,
                 test_start_date,
                 test_end_date,
                 stage,
                 static_attributes_names=[], sequence_length=270,
                 x_mins=None, x_maxs=None, y_mean=None, y_std=None, use_Caravan_dataset=True):
        self.sequence_length = sequence_length
        self.dynamic_data_folder = dynamic_data_folder
        self.static_data_file_caravan = static_data_file_caravan
        self.static_data_file_hydroatlas = static_data_file_hydroatlas
        self.list_static_attributes_names = static_attributes_names
        self.list_dynamic_attributes_names = dynamic_attributes_names
        self.discharge_str = discharge_str
        self.train_start_date = train_start_date
        self.train_end_date = train_end_date
        self.validation_start_date = validation_start_date
        self.validation_end_date = validation_end_date
        self.test_start_date = test_start_date
        self.test_end_date = test_end_date
        self.stage = stage
        self.df_attr, self.list_stations_static = self.read_static_attributes()
        self.use_Caravan_dataset = use_Caravan_dataset
        self.prefix_dynamic_data_file = "us_" if use_Caravan_dataset else "data24_"
        list_stations_repeated, X_data_list, y_data_list = self.read_all_dynamic_data_files(dynamic_data_folder=
                                                                                            dynamic_data_folder)
        self.X_data = np.concatenate(X_data_list)
        self.y_data = np.concatenate(y_data_list)
        self.y_std = y_std if y_std is not None else self.y_data.std()
        self.y_mean = y_mean if y_mean is not None else self.y_data.mean()
        self.x_max = x_maxs if x_maxs is not None else self.X_data.max(axis=0)
        self.x_min = x_mins if x_mins is not None else self.X_data.min(axis=0)
        self.X_data = (self.X_data - self.x_min) / ((self.x_max - self.x_min) + (10 ** -10))
        self.y_data = (self.y_data - self.y_mean) / self.y_std
        self.list_stations_repeated = list_stations_repeated

    def __len__(self):
        return self.calculate_dataset_length(list(set(self.list_stations_repeated)))

    def __getitem__(self, index) -> T_co:
        X_data_tensor = torch.tensor(self.X_data[index: index + self.sequence_length]).to(torch.float32)
        y_data_tensor = torch.tensor(self.y_data[index + self.sequence_length]).to(torch.float32)
        station_id = self.list_stations_repeated[index + self.sequence_length]
        return station_id, X_data_tensor, y_data_tensor

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

    def read_all_dynamic_data_files(self, dynamic_data_folder):
        list_stations_repeated = []
        X_data_list = []
        y_data_list = []
        all_station_files = [f for f in listdir(self.dynamic_data_folder) if isfile(join(dynamic_data_folder, f))]
        for station_file_name in all_station_files:
            station_id = re.search(f"{self.prefix_dynamic_data_file}(\\d+)\\.csv", station_file_name).group(1)
            if station_id in self.list_stations_static:
                station_id_repeated, X_data_curr, y_data_curr = self.read_single_station_file(station_id)
                list_stations_repeated.extend(station_id_repeated)
                if X_data_curr.size > 0:
                    X_data_list.append(X_data_curr)
                    y_data_list.append(y_data_curr)
        return list_stations_repeated, X_data_list, y_data_list

    def read_single_station_file(self, station_id):
        station_data_file = Path(self.dynamic_data_folder) / f"{self.prefix_dynamic_data_file}{station_id}.csv"
        df_dynamic_data = pd.read_csv(station_data_file)
        df_dynamic_data = self.read_and_filter_dynamic_data(df_dynamic_data)
        y_data = df_dynamic_data[self.discharge_str].to_numpy().flatten()
        X_data = df_dynamic_data[self.list_dynamic_attributes_names].to_numpy()
        if X_data.size == 0 or y_data.size == 0:
            return np.array([]), np.array([])
        X_data = X_data.reshape(-1, len(self.list_dynamic_attributes_names))
        y_data = y_data.reshape(-1, 1)
        static_attrib_station = (self.df_attr[self.df_attr["gauge_id"] == station_id]).drop("gauge_id", axis=1) \
            .to_numpy().reshape(1, -1)
        static_attrib_station_rep = static_attrib_station.repeat(X_data.shape[0], axis=0)
        X_data = np.concatenate([X_data, static_attrib_station_rep], axis=1)
        station_id_repeated = [station_id] * X_data.shape[0]
        return station_id_repeated, X_data, y_data

    def read_and_filter_dynamic_data(self, df_dynamic_data):
        df_dynamic_data = df_dynamic_data[self.list_dynamic_attributes_names + [self.discharge_str] + ["date"]]
        df_dynamic_data[self.list_dynamic_attributes_names + [self.discharge_str]] = \
            df_dynamic_data[self.list_dynamic_attributes_names + [self.discharge_str]].applymap(lambda x: float(x))
        df_dynamic_data = df_dynamic_data[df_dynamic_data[self.discharge_str] >= 0]
        df_dynamic_data = df_dynamic_data.dropna()
        df_dynamic_data["date"] = pd.to_datetime(df_dynamic_data.date)
        start_date = self.train_start_date if self.stage == "train" else self.test_start_date
        start_date = datetime.strptime(start_date, "%d/%m/%Y")
        end_date = self.train_end_date if self.stage == "train" else self.test_end_date
        end_date = datetime.strptime(end_date, "%d/%m/%Y")
        df_dynamic_data = df_dynamic_data[
            (df_dynamic_data["date"] >= start_date) & (df_dynamic_data["date"] <= end_date)]
        return df_dynamic_data

    def calculate_dataset_length(self, station_ids):
        count = 0
        for station_id in station_ids:
            station_data_file = Path(self.dynamic_data_folder) / f"{self.prefix_dynamic_data_file}{station_id}.csv"
            df_dynamic_data = pd.read_csv(station_data_file)
            df_dynamic_data = self.read_and_filter_dynamic_data(df_dynamic_data)
            count += (len(df_dynamic_data.index) - self.sequence_length)
        return count

    def get_x_min(self):
        return self.x_min

    def get_x_max(self):
        return self.x_max

    def get_y_std(self):
        return self.y_std

    def get_y_mean(self):
        return self.y_mean
