from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import re
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Pool

matplotlib.use('AGG')

STATIC_ATTRIBUTES_NAMES = ["elev_mean", "slope_mean", "area_gages2", "frac_forest", "lai_max", "lai_diff",
                           "gvf_max", "gvf_diff", "soil_depth_pelletier", "soil_depth_statsgo", "soil_porosity",
                           "soil_conductivity", "max_water_content", "sand_frac", "silt_frac", "clay_frac",
                           "carbonate_rocks_frac", "geol_permeability", "p_mean", "pet_mean", "aridity",
                           "frac_snow", "high_prec_freq", "high_prec_dur", "low_prec_freq", "low_prec_dur"]
DYNAMIC_ATTRIBUTES_NAMES = ["prcp(mm/day)", "srad(w/m2)", "tmax(c)", "tmin(c)", "vp(pa)"]
DISCHARGE_STR = "qobs"
DYNAMIC_DATA_FOLDER = "../data/CAMELS_US/basin_mean_forcing"
STATIC_DATA_FOLDER = "../data/CAMELS_US/camels_attributes_v2.0"
DISCHARGE_DATA_FOLDER = "../data/CAMELS_US/usgs_streamflow"


class Dataset_CAMELS(Dataset):

    def __init__(self,
                 dynamic_data_folder,
                 static_data_folder,
                 discharge_data_folder,
                 dynamic_attributes_names,
                 discharge_str,
                 train_start_date,
                 train_end_date,
                 validation_start_date,
                 validation_end_date,
                 test_start_date,
                 test_end_date,
                 stage,
                 all_stations_ids,
                 static_attributes_names=[],
                 sequence_length=270,
                 x_mins=None,
                 x_maxs=None,
                 y_mean=None,
                 y_std=None):
        self.sequence_length = sequence_length
        self.dynamic_data_folder = dynamic_data_folder
        self.static_data_folder = static_data_folder
        self.discharge_data_folder = discharge_data_folder
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
        self.dict_basin_records_count = {}
        list_stations_repeated, X_data_list, y_data_list = self.read_all_dynamic_and_discharge_data_files(
            all_stations_ids=
            all_stations_ids)
        self.X_data = np.concatenate(X_data_list)
        self.y_data = np.concatenate(y_data_list)
        self.y_std = y_std if y_std is not None else self.y_data.std()
        self.y_mean = y_mean if y_mean is not None else self.y_data.mean()
        self.x_min = x_mins if x_mins is not None else self.X_data.min(axis=0)
        self.x_max = x_maxs if x_maxs is not None else self.X_data.max(axis=0)
        self.X_data = (self.X_data - self.x_min) / ((self.x_max - self.x_min) + 10 ** (-6))
        self.y_data = (self.y_data - self.y_mean) / self.y_std
        self.list_stations_repeated = list_stations_repeated

    def __len__(self):
        return self.calculate_dataset_length()

    def __getitem__(self, index) -> T_co:
        X_data_tensor = torch.tensor(self.X_data[index: index + self.sequence_length]).to(torch.float32)
        y_data_tensor = torch.tensor(self.y_data[index + self.sequence_length]).to(torch.float32)
        station_id = self.list_stations_repeated[index + self.sequence_length]
        return station_id, X_data_tensor, y_data_tensor

    def read_static_attributes(self):
        attributes_path = Path(self.static_data_folder)
        txt_files = attributes_path.glob('camels_*.txt')
        # Read-in attributes into one big dataframe
        dfs = []
        for txt_file in txt_files:
            df_temp = pd.read_csv(txt_file, sep=';', header=0, dtype={'gauge_id': str})
            df_temp = df_temp.set_index('gauge_id')
            dfs.append(df_temp)
        df = pd.concat(dfs, axis=1)
        # convert huc column to double-digit strings
        df['huc'] = df['huc_02'].apply(lambda x: str(x).zfill(2))
        df = df.drop('huc_02', axis=1)
        df = df[self.list_static_attributes_names]
        return df, df.index.to_list()

    def read_all_dynamic_and_discharge_data_files(self, all_stations_ids):
        list_stations_repeated = []
        X_data_list = []
        y_data_list = []
        list_returned = []
        for station_id in all_stations_ids:
            list_returned.append(self.read_single_station_dynamic_and_discharge_file(station_id))
        for station_id_repeated, X_data_curr, y_data_curr in list_returned:
            self.dict_basin_records_count[station_id_repeated[0]] = len(station_id_repeated)
            list_stations_repeated.extend(station_id_repeated)
            if X_data_curr.size > 0:
                X_data_list.append(X_data_curr)
                y_data_list.append(y_data_curr)
        return list_stations_repeated, X_data_list, y_data_list

    def read_single_station_dynamic_and_discharge_file(self, station_id):
        if station_id not in self.list_stations_static:
            return np.array([]), np.array([]), np.array([])
        forcing_path = Path(self.dynamic_data_folder)
        file_path = list(forcing_path.glob(f'**/{station_id}_*_forcing_leap.txt'))
        file_path = file_path[0]
        with open(file_path, 'r') as fp:
            # load area from header
            fp.readline()
            fp.readline()
            area = int(fp.readline())
            # load the dataframe from the rest of the stream
            df_forcing = pd.read_csv(fp, sep='\s+')
            df_forcing["date"] = pd.to_datetime(df_forcing.Year.map(str) + "/"
                                                + df_forcing.Mnth.map(str) + "/"
                                                + df_forcing.Day.map(str),
                                                format="%Y/%m/%d")
            df_forcing = df_forcing.set_index("date")
            discharge_path = Path(self.discharge_data_folder)
            file_path = list(discharge_path.glob(f'**/{station_id}_streamflow_qc.txt'))
            file_path = file_path[0]
            col_names = ['basin', 'Year', 'Mnth', 'Day', 'QObs', 'flag']
            df_discharge = pd.read_csv(file_path, sep='\s+', header=None, names=col_names)
            df_discharge["date"] = pd.to_datetime(df_discharge.Year.map(str) + "/"
                                                  + df_discharge.Mnth.map(str) + "/"
                                                  + df_discharge.Day.map(str),
                                                  format="%Y/%m/%d")
            df_discharge = df_discharge.set_index("date")
            # normalize discharge from cubic feet per second to mm per day
            df_discharge.QObs = 28316846.592 * df_discharge.QObs * 86400 / (area * 10 ** 6)
            df_forcing = df_forcing.drop(columns=['Year', 'Mnth', 'Day'])
            df_discharge = df_discharge.drop(columns=['Year', 'Mnth', 'Day'])
            df_dynamic_data = df_forcing.join(df_discharge, on="date")
            df_dynamic_data.columns = map(str.lower, df_dynamic_data.columns)
            df_dynamic_data = self.read_and_filter_dynamic_data(df_dynamic_data)
            y_data = df_dynamic_data[self.discharge_str].to_numpy().flatten()
            X_data = df_dynamic_data[self.list_dynamic_attributes_names].to_numpy()
            if X_data.size == 0 or y_data.size == 0:
                return np.array([]), np.array([]), np.array([])
            X_data = X_data.reshape(-1, len(self.list_dynamic_attributes_names))
            y_data = y_data.reshape(-1, 1)
            static_attrib_station = (self.df_attr[self.df_attr.index == station_id]).to_numpy().reshape(1, -1)
            static_attrib_station_rep = static_attrib_station.repeat(X_data.shape[0], axis=0)
            X_data = np.concatenate([X_data, static_attrib_station_rep], axis=1)
            station_id_repeated = [station_id] * X_data.shape[0]
            print(f"finished with station id (basin): {station_id}")
            return station_id_repeated, X_data, y_data

    def read_and_filter_dynamic_data(self, df_dynamic_data):
        df_dynamic_data = df_dynamic_data[self.list_dynamic_attributes_names + [self.discharge_str]].copy()
        df_dynamic_data[self.list_dynamic_attributes_names] = \
            df_dynamic_data[self.list_dynamic_attributes_names].astype(float)
        df_dynamic_data.loc[self.discharge_str] = df_dynamic_data[self.discharge_str].apply(lambda x: float(x))
        df_dynamic_data = df_dynamic_data[df_dynamic_data[self.discharge_str] >= 0]
        df_dynamic_data = df_dynamic_data.dropna()
        start_date = self.train_start_date if self.stage == "train" else self.test_start_date
        start_date = datetime.strptime(start_date, "%d/%m/%Y")
        end_date = self.train_end_date if self.stage == "train" else self.test_end_date
        end_date = datetime.strptime(end_date, "%d/%m/%Y")
        df_dynamic_data = df_dynamic_data[
            (df_dynamic_data.index >= start_date) & (df_dynamic_data.index <= end_date)]
        return df_dynamic_data

    def calculate_dataset_length(self):
        count = 0
        for key in self.dict_basin_records_count.keys():
            count += (self.dict_basin_records_count[key] - self.sequence_length)
        return count

    def set_sequence_length(self, sequence_length):
        self.sequence_length = sequence_length
        self.calculate_dataset_length()

    def get_x_mins(self):
        return self.x_min

    def get_x_maxs(self):
        return self.x_max

    def get_y_std(self):
        return self.y_std

    def get_y_mean(self):
        return self.y_mean
