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

matplotlib.use("AGG")

STATIC_ATTRIBUTES_NAMES = [
    "elev_mean",
    "slope_mean",
    "area_gages2",
    "frac_forest",
    "lai_max",
    "lai_diff",
    "gvf_max",
    "gvf_diff",
    "soil_depth_pelletier",
    "soil_depth_statsgo",
    "soil_porosity",
    "soil_conductivity",
    "max_water_content",
    "sand_frac",
    "silt_frac",
    "clay_frac",
    "carbonate_rocks_frac",
    "geol_permeability",
    "p_mean",
    "pet_mean",
    "aridity",
    "frac_snow",
    "high_prec_freq",
    "high_prec_dur",
    "low_prec_freq",
    "low_prec_dur",
]
DYNAMIC_ATTRIBUTES_NAMES = [
    "prcp(mm/day)",
    "srad(w/m2)",
    "tmax(c)",
    "tmin(c)",
    "vp(pa)",
]
DISCHARGE_STR = "qobs"
DYNAMIC_DATA_FOLDER = "../data/CAMELS_US/basin_mean_forcing"
STATIC_DATA_FOLDER = "../data/CAMELS_US/camels_attributes_v2.0"
DISCHARGE_DATA_FOLDER = "../data/CAMELS_US/usgs_streamflow"


class Dataset_CAMELS(Dataset):
    def __init__(
            self,
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
            x_mean_dict=None,
            x_std_dict=None,
            y_mean=None,
            y_std=None
    ):
        self.x_mean_dict = x_mean_dict if x_mean_dict is not None else {}
        self.x_std_dict = x_std_dict if x_std_dict is not None else {}
        self.y_mean_dict = {}
        self.y_std_dict = {}
        self.y_mean = y_mean if y_mean is not None else None
        self.y_std = y_std if y_std is not None else None
        self.sequence_length = sequence_length
        self.dynamic_data_folder = dynamic_data_folder
        self.static_data_folder = static_data_folder
        self.discharge_data_folder = discharge_data_folder
        self.list_static_attributes_names = sorted(static_attributes_names)
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
        (self.dict_station_id_to_data,
         x_means,
         x_stds,
         y_mean,
         y_std
         ) = self.read_all_dynamic_and_discharge_data_files(all_stations_ids=all_stations_ids)

        self.y_mean = y_mean if self.y_mean is None else self.y_mean
        self.y_std = y_std if self.y_std is None else self.y_std

        self.dataset_length, self.lookup_table = self.create_look_table()
        x_data_mean_dynamic = x_means[:(len(self.list_dynamic_attributes_names))]
        x_data_std_dynamic = x_stds[:(len(self.list_dynamic_attributes_names))]

        x_data_mean_static = self.df_attr[self.list_static_attributes_names].mean().to_numpy()
        x_data_std_static = self.df_attr[self.list_static_attributes_names].std().to_numpy()

        for key in self.dict_station_id_to_data.keys():
            current_x_data = self.dict_station_id_to_data[key][0]
            current_y_data = self.dict_station_id_to_data[key][1]

            current_x_data[:, :(len(self.list_dynamic_attributes_names))] = \
                (current_x_data[:, :(len(self.list_dynamic_attributes_names))] - x_data_mean_dynamic) / \
                (x_data_std_dynamic + (10 ** (-6)))

            current_x_data[:, (len(self.list_dynamic_attributes_names)):] = \
                (current_x_data[:, (len(self.list_dynamic_attributes_names)):] - x_data_mean_static) / \
                (x_data_std_static + (10 ** (-6)))

            current_y_data = (current_y_data - self.y_mean) / (self.y_std + (10 ** (-6)))

            self.dict_station_id_to_data[key] = (current_x_data, current_y_data)

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, index) -> T_co:
        next_basin = self.lookup_table[index]
        if self.current_basin != next_basin:
            self.current_basin = next_basin
            self.inner_index_in_data_of_basin = 0
        X_data, y_data = self.dict_station_id_to_data[self.current_basin]
        X_data_tensor = torch.tensor(
            X_data[self.inner_index_in_data_of_basin: self.inner_index_in_data_of_basin + self.sequence_length]
        ).to(torch.float32)
        y_data_tensor = torch.tensor(y_data[self.inner_index_in_data_of_basin + self.sequence_length]).to(
            torch.float32
        )
        self.inner_index_in_data_of_basin += 1
        return self.current_basin, X_data_tensor, y_data_tensor

    def read_static_attributes(self):
        attributes_path = Path(self.static_data_folder)
        txt_files = attributes_path.glob("camels_*.txt")
        # Read-in attributes into one big dataframe
        dfs = []
        for txt_file in txt_files:
            df_temp = pd.read_csv(txt_file, sep=";", header=0, dtype={"gauge_id": str})
            df_temp = df_temp.set_index("gauge_id")
            dfs.append(df_temp)
        df = pd.concat(dfs, axis=1)
        # convert huc column to double-digit strings
        df["huc"] = df["huc_02"].apply(lambda x: str(x).zfill(2))
        df = df.drop("huc_02", axis=1)
        df = df[self.list_static_attributes_names]
        return df, df.index.to_list()

    def read_all_dynamic_and_discharge_data_files(self, all_stations_ids):
        cumm_m_x = 0
        cumm_s_x = 0
        cumm_m_y = 0
        cumm_s_y = 0
        count_of_samples = 0
        dict_station_id_to_data = {}
        for station_id in all_stations_ids:
            X_data, y_data = self.read_single_station_dynamic_and_discharge_file(station_id)
            if len(X_data) == 0 or len(y_data) == 0:
                continue
            dict_station_id_to_data[station_id] = (X_data, y_data)
            prev_mean_x = cumm_m_x
            count_of_samples = count_of_samples + (len(y_data))
            cumm_m_x = cumm_m_x + (
                    (X_data[:, :len(self.list_dynamic_attributes_names)] - cumm_m_x) / count_of_samples).sum(
                axis=0)
            cumm_s_x = cumm_s_x + ((X_data[:, :len(self.list_dynamic_attributes_names)] - cumm_m_x) * (
                    X_data[:, :len(self.list_dynamic_attributes_names)] - prev_mean_x)).sum(axis=0)

            prev_mean_y = cumm_m_y
            cumm_m_y = cumm_m_y + ((y_data[:] - cumm_m_y) / count_of_samples).sum(axis=0)
            cumm_s_y = cumm_s_y + ((y_data[:] - cumm_m_y) * (y_data[:] - prev_mean_y)).sum(axis=0)

        std_x = np.sqrt(cumm_s_x / (count_of_samples - 1))
        std_y = np.sqrt(cumm_s_y / (count_of_samples - 1))
        return dict_station_id_to_data, cumm_m_x, std_x, cumm_m_y.item(), std_y.item()

    def read_single_station_dynamic_and_discharge_file(self, station_id):
        if station_id not in self.list_stations_static:
            return np.array([]), np.array([]), np.array([])
        forcing_path = Path(self.dynamic_data_folder)
        file_path = list(forcing_path.glob(f"**/{station_id}_*_forcing_leap.txt"))
        file_path = file_path[0]
        with open(file_path, "r") as fp:
            # load area from header
            fp.readline()
            fp.readline()
            area = int(fp.readline())
            # load the dataframe from the rest of the stream
            df_forcing = pd.read_csv(fp, sep="\s+")
            df_forcing["date"] = pd.to_datetime(
                df_forcing.Year.map(str)
                + "/"
                + df_forcing.Mnth.map(str)
                + "/"
                + df_forcing.Day.map(str),
                format="%Y/%m/%d",
            )
            df_forcing = df_forcing.set_index("date")
            discharge_path = Path(self.discharge_data_folder)
            file_path = list(discharge_path.glob(f"**/{station_id}_streamflow_qc.txt"))
            file_path = file_path[0]
            col_names = ["basin", "Year", "Mnth", "Day", "QObs", "flag"]
            df_discharge = pd.read_csv(
                file_path, sep="\s+", header=None, names=col_names
            )
            df_discharge["date"] = pd.to_datetime(
                df_discharge.Year.map(str)
                + "/"
                + df_discharge.Mnth.map(str)
                + "/"
                + df_discharge.Day.map(str),
                format="%Y/%m/%d",
            )
            df_discharge = df_discharge.set_index("date")
            # normalize discharge from cubic feet per second to mm per day
            df_discharge.QObs = (
                    28316846.592 * df_discharge.QObs * 86400 / (area * 10 ** 6)
            )
            df_forcing = df_forcing.drop(columns=["Year", "Mnth", "Day"])
            df_discharge = df_discharge.drop(columns=["Year", "Mnth", "Day"])
            df_dynamic_data = df_forcing.join(df_discharge, on="date")
            df_dynamic_data.columns = map(str.lower, df_dynamic_data.columns)
            df_dynamic_data = self.read_and_filter_dynamic_data(df_dynamic_data)

            y_data = df_dynamic_data[self.discharge_str].to_numpy().flatten()
            X_data = df_dynamic_data[self.list_dynamic_attributes_names].to_numpy()
            if X_data.size == 0 or y_data.size == 0:
                return np.array([]), np.array([]), np.array([])
            X_data = X_data.reshape(-1, len(self.list_dynamic_attributes_names))
            y_data = y_data.reshape(-1, 1)
            static_attrib_station = (
                (self.df_attr[self.df_attr.index == station_id])
                .to_numpy()
                .reshape(1, -1)
            )
            static_attrib_station_rep = static_attrib_station.repeat(
                X_data.shape[0], axis=0
            )
            if station_id not in self.x_mean_dict.keys():
                self.x_mean_dict[station_id] = X_data.mean(axis=0)
            if station_id not in self.x_std_dict.keys():
                self.x_std_dict[station_id] = X_data.std(axis=0)
            X_data = np.concatenate([X_data, static_attrib_station_rep], axis=1)
            # print(f"finished with station id (basin): {station_id}")
            if station_id not in self.y_mean_dict.keys():
                self.y_mean_dict[station_id] = torch.tensor(y_data.mean(axis=0))
            if station_id not in self.y_std_dict.keys():
                self.y_std_dict[station_id] = torch.tensor(y_data.std(axis=0))
            # y_data -= (self.y_mean_dict[station_id].numpy())
            # y_data /= (self.y_std_dict[station_id].numpy())
            return X_data, y_data

    def read_and_filter_dynamic_data(self, df_dynamic_data):
        df_dynamic_data = df_dynamic_data[
            self.list_dynamic_attributes_names + [self.discharge_str]
            ].copy()
        df_dynamic_data[self.list_dynamic_attributes_names] = df_dynamic_data[
            self.list_dynamic_attributes_names
        ].astype(float)
        df_dynamic_data.loc[self.discharge_str] = df_dynamic_data[
            self.discharge_str
        ].apply(lambda x: float(x))
        df_dynamic_data = df_dynamic_data[df_dynamic_data[self.discharge_str] >= 0]
        df_dynamic_data = df_dynamic_data.dropna()
        start_date = (
            self.train_start_date if self.stage == "train" else self.test_start_date
        )
        start_date = datetime.strptime(start_date, "%d/%m/%Y")
        end_date = self.train_end_date if self.stage == "train" else self.test_end_date
        end_date = datetime.strptime(end_date, "%d/%m/%Y")
        df_dynamic_data = df_dynamic_data[
            (df_dynamic_data.index >= start_date) & (df_dynamic_data.index <= end_date)
            ]
        return df_dynamic_data

    def calculate_dataset_length(self):
        return self.dataset_length

    def create_look_table(self):
        self.inner_index_in_data_of_basin = 0
        lookup_table_basins = {}
        length_of_dataset = 0
        self.current_basin = list(self.dict_station_id_to_data.keys())[0]
        for key in self.dict_station_id_to_data.keys():
            for _ in range(len(self.dict_station_id_to_data[key][0]) - self.sequence_length):
                lookup_table_basins[length_of_dataset] = key
                length_of_dataset += 1
        return length_of_dataset, lookup_table_basins

    def set_sequence_length(self, sequence_length):
        self.sequence_length = sequence_length
        self.dataset_length, self.lookup_table = self.create_look_table()

    def get_x_mins(self):
        return self.x_mins_dict

    def get_x_maxs(self):
        return self.x_maxs_dict

    def get_y_std(self):
        return self.y_std

    def get_y_mean(self):
        return self.y_mean
