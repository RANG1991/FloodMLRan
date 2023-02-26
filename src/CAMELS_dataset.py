from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
import matplotlib
import psutil
import gc
import codecs
import json
import pickle
import os
from tqdm import tqdm
import sys
from os.path import isfile, join

matplotlib.use("AGG")

STATIC_ATTRIBUTES_NAMES = [
    "p_mean",
    "pet_mean",
    "aridity",
    "p_seasonality",
    "frac_snow",
    "high_prec_freq",
    "high_prec_dur",
    "low_prec_freq",
    "low_prec_dur",
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
    "geol_permeability"
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

FOLDER_WITH_BASINS_PICKLES = "../data/CAMELS_US/pickled_basins_data"

JSON_FILE_MEAN_STD_COUNT = f"{FOLDER_WITH_BASINS_PICKLES}/mean_std_count_of_data.json"

X_MEAN_DICT_FILE = f"{FOLDER_WITH_BASINS_PICKLES}/x_mean_dict.pkl"

X_STD_DICT_FILE = f"{FOLDER_WITH_BASINS_PICKLES}/x_std_dict.pkl"

Y_MEAN_DICT_FILE = f"{FOLDER_WITH_BASINS_PICKLES}/y_mean_dict.pkl"

Y_STD_DICT_FILE = f"{FOLDER_WITH_BASINS_PICKLES}/y_std_dict.pkl"


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
            specific_model_type,
            create_new_files,
            static_attributes_names=[],
            sequence_length=270,
            x_means=None,
            x_stds=None,
            y_mean=None,
            y_std=None
    ):
        self.x_mean_dict = self.read_pickle_if_exists(f"{X_MEAN_DICT_FILE}")
        self.x_std_dict = self.read_pickle_if_exists(f"{X_STD_DICT_FILE}")
        self.x_means = x_means if x_means is not None else None
        self.x_stds = x_stds if x_stds is not None else None
        self.y_mean_dict = self.read_pickle_if_exists(f"{Y_MEAN_DICT_FILE}")
        self.y_std_dict = self.read_pickle_if_exists(f"{Y_STD_DICT_FILE}")
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
        self.specific_model_type = specific_model_type
        self.df_attr, self.list_stations_static = self.read_static_attributes()
        self.all_station_ids = sorted(list(set(all_stations_ids).intersection(set(self.list_stations_static))))
        (dict_station_id_to_data,
         x_means,
         x_stds,
         y_mean,
         y_std
         ) = self.read_all_dynamic_data_files(all_stations_ids=all_stations_ids,
                                              create_new_files=create_new_files)

        self.save_pickle_if_not_exists(f"{X_MEAN_DICT_FILE}", self.x_mean_dict, force=True)
        self.save_pickle_if_not_exists(f"{X_STD_DICT_FILE}", self.x_std_dict, force=True)
        self.save_pickle_if_not_exists(f"{Y_MEAN_DICT_FILE}", self.y_mean_dict, force=True)
        self.save_pickle_if_not_exists(f"{Y_STD_DICT_FILE}", self.y_std_dict, force=True)

        self.y_mean = y_mean if stage == "train" else self.y_mean
        self.y_std = y_std if stage == "train" else self.y_std
        self.x_means = x_means if stage == "train" else self.x_means
        self.x_stds = x_stds if stage == "train" else self.x_stds

        x_data_mean_dynamic = self.x_means[:(len(self.list_dynamic_attributes_names))]
        x_data_std_dynamic = self.x_stds[:(len(self.list_dynamic_attributes_names))]

        x_data_mean_static = self.df_attr[self.list_static_attributes_names].mean().to_numpy()
        x_data_std_static = self.df_attr[self.list_static_attributes_names].std().to_numpy()

        if create_new_files or len(dict_station_id_to_data) > 0:
            for key in dict_station_id_to_data.keys():
                current_x_data = dict_station_id_to_data[key]["x_data"]
                current_y_data = dict_station_id_to_data[key]["y_data"]
                current_x_data[:, :(len(self.list_dynamic_attributes_names))] = \
                    (current_x_data[:, :(len(self.list_dynamic_attributes_names))] - x_data_mean_dynamic) / \
                    (x_data_std_dynamic + (10 ** (-6)))

                current_x_data[:, (len(self.list_dynamic_attributes_names)):] = \
                    (current_x_data[:, (len(self.list_dynamic_attributes_names)):] - x_data_mean_static) / \
                    (x_data_std_static + (10 ** (-6)))
                current_y_data = (current_y_data - self.y_mean) / (self.y_std + (10 ** (-6)))
                dict_curr_basin = {"x_data": current_x_data, "y_data": current_y_data}
                dict_station_id_to_data[key] = dict_curr_basin
                with open(f"{FOLDER_WITH_BASINS_PICKLES}/{key}_{self.stage}.pkl",
                          'wb') as f:
                    pickle.dump(dict_curr_basin, f)
        dict_station_id_to_data_from_file = self.load_basins_dicts_from_pickles()
        self.all_station_ids = list(dict_station_id_to_data_from_file.keys())
        self.dataset_length, self.lookup_table = self.create_look_table(dict_station_id_to_data_from_file)
        del dict_station_id_to_data_from_file

    def __len__(self):
        return self.dataset_length

    @staticmethod
    def crop_or_pad_precip_spatial(X_data_single_basin, max_width, max_height):
        max_width_right = int(max_width / 2)
        max_width_left = math.ceil(max_width / 2)
        max_height_right = int(max_height / 2)
        max_height_left = math.ceil(max_height / 2)
        if X_data_single_basin.shape[1] > max_width:
            start = X_data_single_basin.shape[1] // 2 - (max_width // 2)
            X_data_single_basin = X_data_single_basin[:, start:start + max_width, :]
        else:
            X_data_single_basin = np.pad(X_data_single_basin,
                                         ((0, 0),
                                          (max_width_right - int(X_data_single_basin.shape[1] / 2),
                                           max_width_left - math.ceil(X_data_single_basin.shape[1] / 2)),
                                          (0, 0)),
                                         "constant",
                                         constant_values=0)
        if X_data_single_basin.shape[2] > max_height:
            start = X_data_single_basin.shape[2] // 2 - (max_height // 2)
            X_data_single_basin = X_data_single_basin[:, start:start + max_height, :]
        else:
            X_data_single_basin = np.pad(X_data_single_basin,
                                         ((0, 0),
                                          (0, 0),
                                          (max_height_right - int(X_data_single_basin.shape[2] / 2),
                                           max_height_left - math.ceil(X_data_single_basin.shape[2] / 2))),
                                         "constant",
                                         constant_values=0)
        return X_data_single_basin

    def __getitem__(self, index) -> T_co:
        basin_id, inner_ind = self.lookup_table[index]
        with open(f"{FOLDER_WITH_BASINS_PICKLES}/{basin_id}_{self.stage}{self.suffix_pickle_file}.pkl",
                  'rb') as f:
            dict_curr_basin = pickle.load(f)
        X_data_tensor_spatial = torch.tensor([])
        if self.specific_model_type.lower() == "lstm" or self.specific_model_type.lower() == "transformer_lstm":
            X_data, y_data = dict_curr_basin["x_data"], dict_curr_basin["y_data"]
            X_data_tensor_non_spatial = torch.tensor(
                X_data[inner_ind: inner_ind + self.sequence_length]
            ).to(torch.float32)
        elif self.specific_model_type.lower() == "conv":
            X_data, X_data_spatial, y_data = \
                dict_curr_basin["x_data"], dict_curr_basin["x_data_spatial"], dict_curr_basin["y_data"]
            X_data_tensor_non_spatial = torch.tensor(
                X_data[inner_ind: inner_ind + self.sequence_length - self.sequence_length_spatial]
            ).to(torch.float32)
            X_data_tensor_spatial = torch.tensor(
                X_data_spatial[
                inner_ind + self.sequence_length - self.sequence_length_spatial: inner_ind + self.sequence_length]
            ).to(torch.float32)
        elif self.specific_model_type.lower() == "cnn":
            X_data, X_data_spatial, y_data = \
                dict_curr_basin["x_data"], dict_curr_basin["x_data_spatial"], dict_curr_basin["y_data"]
            X_data_tensor_non_spatial = torch.tensor(
                X_data[inner_ind: inner_ind + self.sequence_length]
            ).to(torch.float32)
            X_data_tensor_spatial = torch.tensor(
                X_data_spatial[
                inner_ind + self.sequence_length - self.sequence_length_spatial: inner_ind + self.sequence_length]
            ).to(torch.float32)
        elif self.specific_model_type.lower() == "transformer_seq2seq":
            X_data, y_data = dict_curr_basin["x_data"], dict_curr_basin["y_data"]
            X_data_tensor_non_spatial = torch.tensor(
                X_data[inner_ind: inner_ind + self.sequence_length]
            ).to(torch.float32)
        else:
            X_data, X_data_spatial, y_data = dict_curr_basin["x_data"], dict_curr_basin["x_data_spatial"], \
                dict_curr_basin["y_data"]
            X_data_tensor_non_spatial = torch.tensor(
                X_data[inner_ind: inner_ind + self.sequence_length]
            ).to(torch.float32)
            X_data_tensor_spatial = torch.tensor(X_data_spatial[inner_ind: inner_ind + self.sequence_length]).to(
                torch.float32)
        if self.specific_model_type.lower() == "transformer_seq2seq":
            y_data_tensor = torch.tensor(
                y_data[inner_ind + 1: inner_ind + self.sequence_length + 1]
            ).to(torch.float32).squeeze()
        else:
            y_data_tensor = torch.tensor(y_data[inner_ind + self.sequence_length - 1]
                                         ).to(torch.float32).squeeze()
        return self.y_std_dict[basin_id], basin_id, X_data_tensor_non_spatial, X_data_tensor_spatial, y_data_tensor

    @staticmethod
    def read_pickle_if_exists(pickle_file_name):
        dict_obj = {}
        if os.path.exists(pickle_file_name):
            with open(pickle_file_name, "rb") as f:
                dict_obj = pickle.load(f)
        return dict_obj

    @staticmethod
    def save_pickle_if_not_exists(pickle_file_name, obj_to_save, force=False):
        if not os.path.exists(pickle_file_name) or force:
            with open(pickle_file_name, "wb") as f:
                pickle.dump(obj_to_save, f)

    def check_is_valid_station_id(self, station_id, create_new_files):
        return (station_id in self.all_station_ids
                and (not os.path.exists(
                    f"{FOLDER_WITH_BASINS_PICKLES}/{station_id}_{self.stage}.pkl")
                     or any([not os.path.exists(
                            f"{JSON_FILE_MEAN_STD_COUNT}_{self.stage}"),
                             station_id not in self.x_mean_dict,
                             station_id not in self.x_std_dict,
                             station_id not in self.y_mean_dict,
                             station_id not in self.y_std_dict,
                             create_new_files])))

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

    def read_all_dynamic_data_files(self, all_stations_ids, create_new_files):
        if os.path.exists(f"{JSON_FILE_MEAN_STD_COUNT}_{self.stage}") and not create_new_files:
            obj_text = codecs.open(f"{JSON_FILE_MEAN_STD_COUNT}_{self.stage}", 'r',
                                   encoding='utf-8').read()
            json_obj = json.loads(obj_text)
            cumm_m_x = np.array(json_obj["cumm_m_x"])
            cumm_s_x = np.array(json_obj["cumm_s_x"])
            cumm_m_y = float(json_obj["cumm_m_y"])
            cumm_s_y = float(json_obj["cumm_s_y"])
            count_of_samples = int(json_obj["count_of_samples"])
        else:
            cumm_m_x = 0
            cumm_s_x = 0
            cumm_m_y = 0
            cumm_s_y = 0
            count_of_samples = 0
        dict_station_id_to_data = {}
        pbar = tqdm(all_stations_ids, file=sys.stdout)
        pbar.set_description(f"processing basins - {self.stage}")
        for station_id in pbar:
            print('RAM Used (GB):', psutil.virtual_memory()[3] / 1000000000)
            if self.check_is_valid_station_id(station_id, create_new_files=create_new_files):
                X_data_non_spatial, y_data = self.read_single_station_dynamic_and_discharge_file(station_id)
                if len(X_data_non_spatial) == 0 or len(y_data) == 0:
                    del X_data_non_spatial
                    del y_data
                    continue
                prev_mean_x = cumm_m_x
                count_of_samples = count_of_samples + (len(y_data))
                cumm_m_x = cumm_m_x + (
                        (X_data_non_spatial[:,
                         :len(self.list_dynamic_attributes_names)] - cumm_m_x) / count_of_samples).sum(
                    axis=0)
                cumm_s_x = cumm_s_x + ((X_data_non_spatial[:, :len(self.list_dynamic_attributes_names)] - cumm_m_x) * (
                        X_data_non_spatial[:, :len(self.list_dynamic_attributes_names)] - prev_mean_x)).sum(axis=0)
                prev_mean_y = cumm_m_y
                cumm_m_y = cumm_m_y + ((y_data[:] - cumm_m_y) / count_of_samples).sum(axis=0).item()
                cumm_s_y = cumm_s_y + ((y_data[:] - cumm_m_y) * (y_data[:] - prev_mean_y)).sum(axis=0).item()
                dict_station_id_to_data[station_id] = {"x_data": X_data_non_spatial, "y_data": y_data}
            else:
                print(f"station with id: {station_id} has no valid file or the file already exists")
        gc.collect()
        std_x = np.sqrt(cumm_s_x / (count_of_samples - 1))
        std_y = np.sqrt(cumm_s_y / (count_of_samples - 1)).item()
        with codecs.open(f"{JSON_FILE_MEAN_STD_COUNT}_{self.stage}", 'w',
                         encoding='utf-8') as json_file:
            json_obj = {
                "cumm_m_x": cumm_m_x.tolist(),
                "cumm_s_x": cumm_s_x.tolist(),
                "cumm_m_y": cumm_m_y,
                "cumm_s_y": cumm_s_y,
                "count_of_samples": count_of_samples
            }
            json.dump(json_obj, json_file, separators=(',', ':'), sort_keys=True, indent=4)
        return dict_station_id_to_data, cumm_m_x, std_x, cumm_m_y, std_y

    def read_single_station_dynamic_and_discharge_file(self, station_id):
        if station_id not in self.all_station_ids:
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

    def create_look_table(self, dict_station_id_to_data):
        lookup_table_basins = {}
        length_of_dataset = 0
        for key in dict_station_id_to_data.keys():
            for ind in range(len(dict_station_id_to_data[key]["x_data"]) - self.sequence_length):
                lookup_table_basins[length_of_dataset] = (key, ind)
                length_of_dataset += 1
        return length_of_dataset, lookup_table_basins

    def load_basins_dicts_from_pickles(self):
        dict_station_id_to_data = {}
        for basin_id in self.all_station_ids:
            file_name = join(f"{FOLDER_WITH_BASINS_PICKLES}", f"{basin_id}_{self.stage}.pkl")
            if os.path.exists(file_name):
                with open(file_name, "rb") as f:
                    pickled_data = pickle.load(f)
                    dict_station_id_to_data[basin_id] = pickled_data
        return dict_station_id_to_data

    def set_sequence_length(self, sequence_length):
        self.sequence_length = sequence_length
        dict_station_id_to_data = self.load_basins_dicts_from_pickles()
        self.dataset_length, self.lookup_table = self.create_look_table(dict_station_id_to_data)

    def get_x_means(self):
        return self.x_means

    def get_x_stds(self):
        return self.x_stds

    def get_y_std(self):
        return self.y_std

    def get_y_mean(self):
        return self.y_mean
