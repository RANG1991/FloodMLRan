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
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Pool

ATTRIBUTES_TO_TEXT_DESC = {"p_mean": "Mean daily precipitation",
                           "pet_mean": "Mean daily potential evapotranspiration",
                           "aridity": "Ratio of mean PET to mean precipitation",
                           "seasonality": "Seasonality and timing of precipitation",
                           "frac_snow": "Fraction of precipitation falling\non days with temperatures below 0 ◦C",
                           "high_prec_freq": "Frequency of high-precipitation days",
                           "high_prec_dur": "Average duration of high-precipitation events",
                           "low_prec_freq": "Frequency of dry days",
                           "low_prec_dur": "Average duration of dry periods",
                           "ele_mt_sav": "Catchment mean elevation",
                           "slp_dg_sav": "Catchment mean slope",
                           "basin_area": "Catchment area",
                           "for_pc_sse": "for_pc_sse",
                           "cly_pc_sav": "cly_pc_sav",
                           "slt_pc_sav": "slt_pc_sav",
                           "snd_pc_sav": "snd_pc_sav",
                           "soc_th_sav": "Organic carbon content in soil",
                           "total_precipitation_sum": "total_precipitation_sum",
                           "temperature_2m_min": "temperature_2m_min",
                           "temperature_2m_max": "temperature_2m_max",
                           "potential_evaporation_sum": "potential_evaporation_sum",
                           "surface_net_solar_radiation_mean": "surface_net_solar_radiation_mean"}


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
                 all_station_files,
                 static_attributes_names=[],
                 sequence_length=270,
                 x_means=None, x_stds=None,
                 y_mean=None, y_std=None,
                 use_Caravan_dataset=True):
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
        list_stations_repeated, X_data_list, y_data_list = self.read_all_dynamic_data_files(all_station_files=
                                                                                            all_station_files)
        self.X_data = np.concatenate(X_data_list)
        self.y_data = np.concatenate(y_data_list)
        self.y_std = y_std if y_std is not None else self.y_data.std()
        self.y_mean = y_mean if y_mean is not None else self.y_data.mean()
        self.x_mean = x_means if x_means is not None else self.X_data.mean(axis=0)
        self.x_std = x_stds if x_stds is not None else self.X_data.std(axis=0)
        self.X_data = (self.X_data - self.x_mean) / self.x_std
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
        # maxes = df_attr.drop(columns=['gauge_id']).max(axis=1).to_numpy().reshape(-1, 1)
        # mins = df_attr.drop(columns=['gauge_id']).min(axis=1).to_numpy().reshape(-1, 1)
        df_attr[self.list_static_attributes_names] = df_attr.drop(columns=['gauge_id']).to_numpy()
        return df_attr, df_attr['gauge_id'].values.tolist()

    def read_all_dynamic_data_files(self, all_station_files):
        list_stations_repeated = []
        X_data_list = []
        y_data_list = []
        with Pool(multiprocessing.cpu_count() - 1) as p:
            list_returned = p.map(self.read_single_station_file, all_station_files)
        for station_id_repeated, X_data_curr, y_data_curr in list_returned:
            list_stations_repeated.extend(station_id_repeated)
            if X_data_curr.size > 0:
                X_data_list.append(X_data_curr)
                y_data_list.append(y_data_curr)
        return list_stations_repeated, X_data_list, y_data_list

    def read_single_station_file(self, station_file_name):
        station_id = re.search(f"{self.prefix_dynamic_data_file}(\\d+)\\.csv", station_file_name).group(1)
        if station_id not in self.list_stations_static:
            return np.array([]), np.array([]), np.array([])
        station_data_file = Path(self.dynamic_data_folder) / f"{self.prefix_dynamic_data_file}{station_id}.csv"
        df_dynamic_data = pd.read_csv(station_data_file)
        df_dynamic_data = self.read_and_filter_dynamic_data(df_dynamic_data)
        y_data = df_dynamic_data[self.discharge_str].to_numpy().flatten()
        X_data = df_dynamic_data[self.list_dynamic_attributes_names].to_numpy()
        if X_data.size == 0 or y_data.size == 0:
            return np.array([]), np.array([]), np.array([])
        X_data = X_data.reshape(-1, len(self.list_dynamic_attributes_names))
        y_data = y_data.reshape(-1, 1)
        static_attrib_station = (self.df_attr[self.df_attr["gauge_id"] == station_id]).drop("gauge_id", axis=1) \
            .to_numpy().reshape(1, -1)
        static_attrib_station_rep = static_attrib_station.repeat(X_data.shape[0], axis=0)
        X_data = np.concatenate([X_data, static_attrib_station_rep], axis=1)
        station_id_repeated = [station_id] * X_data.shape[0]
        return station_id_repeated, X_data, y_data

    def read_and_filter_dynamic_data(self, df_dynamic_data):
        df_dynamic_data = df_dynamic_data[self.list_dynamic_attributes_names + [self.discharge_str] + ["date"]].copy()
        df_dynamic_data[self.list_dynamic_attributes_names] = \
            df_dynamic_data[self.list_dynamic_attributes_names].astype(float)
        df_dynamic_data.loc[self.discharge_str] = df_dynamic_data[self.discharge_str].apply(lambda x: float(x))
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

    def create_boxplot_of_entire_dataset(self):
        all_attributes_names = self.list_static_attributes_names + self.list_dynamic_attributes_names
        dict_boxplots_data = {}
        for i in range(self.X_data.shape[1]):
            dict_boxplots_data[ATTRIBUTES_TO_TEXT_DESC[all_attributes_names[i]]] = self.X_data[:, i]
        Dataset_ERA5.create_boxplot_on_data(dict_boxplots_data, plot_title=f"{self.stage}")


    @staticmethod
    def create_boxplot_on_data(dict_boxplots_data, plot_title=""):
        fig, ax = plt.subplots()
        dict_boxplots_data = {k: v for k, v in sorted(dict_boxplots_data.items(), key=lambda item: item[0])}
        box_plots = ax.boxplot(dict_boxplots_data.values(),
                               showfliers=False,
                               positions=list(range(1, len(dict_boxplots_data.keys()) + 1)))
        ax.set_xticks(np.arange(1, len(dict_boxplots_data.keys()) + 1))
        plt.yticks(fontsize=6)
        plt.legend([box_plots['boxes'][i] for i in range(len(dict_boxplots_data.keys()))],
                   [f"{i + 1} :" + k
                    for i, k in enumerate(dict_boxplots_data.keys())],
                   fontsize=6, handlelength=0, handletextpad=0)
        fig.tight_layout()
        curr_datetime = datetime.now()
        curr_datetime_str = curr_datetime.strftime("%d-%m-%Y_%H:%M:%S")
        plt.grid()
        plt.title(f"Box plots data - {plot_title}", fontsize=8)
        plt.savefig(f"../data/images/data_box_plots_{plot_title}" +
                    f"_{curr_datetime_str}" + ".png")


    def set_sequence_length(self, sequence_length):
        self.sequence_length = sequence_length
        self.calculate_dataset_length(list(set(self.list_stations_repeated)))

    def get_x_mean(self):
        return self.x_mean

    def get_x_std(self):
        return self.x_std

    def get_y_std(self):
        return self.y_std

    def get_y_mean(self):
        return self.y_mean
