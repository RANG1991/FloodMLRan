from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
import numpy as np
import torch
import matplotlib
import os
import math
import pickle
from os.path import join
import csv
from glob import glob

matplotlib.use("AGG")


class FloodML_Base_Dataset(Dataset):

    def __init__(self,
                 main_folder,
                 dynamic_data_folder,
                 static_data_folder,
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
                 sequence_length_spatial,
                 specific_model_type="",
                 static_attributes_names=[],
                 sequence_length=270,
                 x_means=None,
                 x_stds=None,
                 y_mean=None,
                 y_std=None,
                 create_new_files=False,
                 limit_size_above_1000=False,
                 use_all_static_attr=False):
        self.main_folder = main_folder
        self.limit_size_above_1000 = limit_size_above_1000
        self.use_all_static_attr = use_all_static_attr
        self.folder_with_basins_pickles = f"{main_folder}/pickled_basins_data"
        self.list_stations_static = []
        self.suffix_pickle_file = "" if specific_model_type.lower() == "lstm" else "_spatial"
        self.x_mean_dict = self.read_pickle_if_exists(
            f"{self.folder_with_basins_pickles}/x_mean_dict.pkl{self.suffix_pickle_file}")
        self.x_std_dict = self.read_pickle_if_exists(
            f"{self.folder_with_basins_pickles}/x_std_dict.pkl{self.suffix_pickle_file}")
        self.x_means = x_means if x_means is not None else None
        self.x_stds = x_stds if x_stds is not None else None
        self.y_mean_dict = self.read_pickle_if_exists(
            f"{self.folder_with_basins_pickles}/y_mean_dict.pkl{self.suffix_pickle_file}")
        self.y_std_dict = self.read_pickle_if_exists(
            f"{self.folder_with_basins_pickles}/y_std_dict.pkl{self.suffix_pickle_file}")
        self.y_mean = y_mean if y_mean is not None else None
        self.y_std = y_std if y_std is not None else None
        self.sequence_length = sequence_length
        self.dynamic_data_folder = dynamic_data_folder
        self.static_data_folder = static_data_folder
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
        self.sequence_length_spatial = sequence_length_spatial
        self.create_new_files = create_new_files
        (self.df_attr,
         self.list_stations_static
         ) = self.read_all_static_attributes(limit_size_above_1000=self.limit_size_above_1000)
        # self.all_station_ids = sorted(list(set(all_stations_ids).intersection(set(self.list_stations_static))))
        self.all_station_ids = self.list_stations_static
        (max_width,
         max_height,
         basin_id_with_maximum_width,
         basin_id_with_maximum_height) = \
            self.get_maximum_width_and_length_of_basin("../data/ERA5/ERA_5_all_data", self.all_station_ids)
        if max_height <= 0 or max_width <= 0:
            raise Exception("max length or max width are not greater than 0")
        self.max_width = max_width
        self.max_height = max_height
        self.max_dim = max(self.max_height, self.max_width)
        (dict_station_id_to_data,
         x_means,
         x_stds,
         y_mean,
         y_std,
         x_spatial_mean,
         x_spatial_std
         ) = self.read_all_dynamic_attributes(all_stations_ids=self.all_station_ids,
                                              specific_model_type=self.specific_model_type,
                                              max_width=self.max_width, max_height=self.max_height,
                                              create_new_files=self.create_new_files)
        self.save_pickle_if_not_exists(
            f"{self.folder_with_basins_pickles}/x_mean_dict.pkl{self.suffix_pickle_file}", self.x_mean_dict,
            force=True)
        self.save_pickle_if_not_exists(
            f"{self.folder_with_basins_pickles}/x_std_dict.pkl{self.suffix_pickle_file}", self.x_std_dict,
            force=True)
        self.save_pickle_if_not_exists(
            f"{self.folder_with_basins_pickles}/y_mean_dict.pkl{self.suffix_pickle_file}", self.y_mean_dict,
            force=True)
        self.save_pickle_if_not_exists(
            f"{self.folder_with_basins_pickles}/y_std_dict.pkl{self.suffix_pickle_file}", self.y_std_dict,
            force=True)

        self.y_mean = y_mean if self.stage == "train" else self.y_mean
        self.y_std = y_std if self.stage == "train" else self.y_std
        self.x_means = x_means if self.stage == "train" else self.x_means
        self.x_stds = x_stds if self.stage == "train" else self.x_stds

        x_data_mean_dynamic = self.x_means[:(len(self.list_dynamic_attributes_names))]
        x_data_std_dynamic = self.x_stds[:(len(self.list_dynamic_attributes_names))]

        x_data_mean_static = self.df_attr[self.list_static_attributes_names].mean().to_numpy()
        x_data_std_static = self.df_attr[self.list_static_attributes_names].std().to_numpy()

        if self.create_new_files or len(dict_station_id_to_data) > 0:
            for key in dict_station_id_to_data.keys():
                current_x_data = dict_station_id_to_data[key]["x_data"]
                current_y_data = dict_station_id_to_data[key]["y_data"]

                indices_features_dynamic_non_spatial = range(0, (len(self.list_dynamic_attributes_names)))

                current_x_data[:, indices_features_dynamic_non_spatial] = \
                    (current_x_data[:, indices_features_dynamic_non_spatial] - x_data_mean_dynamic) / \
                    (x_data_std_dynamic + (10 ** (-6)))

                indices_features_static = range((len(self.list_dynamic_attributes_names)),
                                                (len(self.list_dynamic_attributes_names))
                                                + (len(self.list_static_attributes_names)))

                current_x_data[:, indices_features_static] = \
                    (current_x_data[:, indices_features_static] - x_data_mean_static) / (
                            x_data_std_static + (10 ** (-6)))

                current_y_data = (current_y_data - self.y_mean) / (self.y_std + (10 ** (-6)))

                if (self.specific_model_type.lower() == "lstm" or self.specific_model_type.lower() ==
                        "transformer_seq2seq" or self.specific_model_type.lower() == "transformer_lstm"):
                    dict_curr_basin = {"x_data": current_x_data, "y_data": current_y_data}
                else:
                    current_x_data_spatial = current_x_data[:, ((len(self.list_dynamic_attributes_names))
                                                                + (len(self.list_static_attributes_names))):]
                    indices_all_features_non_spatial = range(0,
                                                             (len(self.list_dynamic_attributes_names))
                                                             + (len(self.list_static_attributes_names)))
                    current_x_data_non_spatial = current_x_data[:, indices_all_features_non_spatial]
                    del current_x_data
                    dict_curr_basin = {"x_data": current_x_data_non_spatial, "y_data": current_y_data,
                                       "x_data_spatial": current_x_data_spatial}
                with open(f"{self.folder_with_basins_pickles}/{key}_{self.stage}{self.suffix_pickle_file}.pkl",
                          'wb') as f:
                    pickle.dump(dict_curr_basin, f)
        dict_station_id_to_data_from_file = self.load_basins_dicts_from_pickles()
        self.all_station_ids = list(dict_station_id_to_data_from_file.keys())
        self.dataset_length, self.lookup_table = self.create_look_table(dict_station_id_to_data_from_file)
        del dict_station_id_to_data

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

    def __len__(self):
        return self.dataset_length

    def calculate_dataset_length(self):
        return self.dataset_length

    def __getitem__(self, index) -> T_co:
        basin_id, inner_ind = self.lookup_table[index]
        with open(f"{self.folder_with_basins_pickles}/{basin_id}_{self.stage}{self.suffix_pickle_file}.pkl",
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

    def create_look_table(self, dict_station_id_to_data):
        lookup_table_basins = {}
        length_of_dataset = 0
        for key in dict_station_id_to_data.keys():
            for ind in range(len(dict_station_id_to_data[key]["x_data"]) - self.sequence_length):
                lookup_table_basins[length_of_dataset] = (key, ind)
                length_of_dataset += 1
        return length_of_dataset, lookup_table_basins

    def check_is_valid_station_id(self, station_id, create_new_files):
        raise NotImplementedError

    def load_basins_dicts_from_pickles(self):
        dict_station_id_to_data = {}
        for basin_id in self.all_station_ids:
            file_name = join(f"{self.folder_with_basins_pickles}/",
                             f"{basin_id}_{self.stage}{self.suffix_pickle_file}.pkl")
            if os.path.exists(file_name):
                with open(file_name, "rb") as f:
                    pickled_data = pickle.load(f)
                    dict_station_id_to_data[basin_id] = pickled_data
        return dict_station_id_to_data

    def set_sequence_length(self, sequence_length):
        self.sequence_length = sequence_length
        dict_station_id_to_data = self.load_basins_dicts_from_pickles()
        self.dataset_length, self.lookup_table = self.create_look_table(dict_station_id_to_data)

    def get_x_stds(self):
        return self.x_stds

    def get_x_means(self):
        return self.x_means

    def get_y_std(self):
        return self.y_std

    def get_y_mean(self):
        return self.y_mean

    def get_maximum_width_and_length_of_basin(self, shape_files_folder, basins_ids):
        WIDTH_LOC_IN_ROW = 2
        HEIGHT_LOC_IN_ROW = 3
        max_height = -1
        max_width = -1
        basin_id_with_maximum_height = -1
        basin_id_with_maximum_width = -1
        file_names = glob(f"{shape_files_folder}/shape_*.csv")
        for file_name in file_names:
            basin_id = file_name.replace(f"{shape_files_folder}/shape", "").replace("_", "").strip(".csv")
            if basin_id in basins_ids and self.check_is_valid_station_id(station_id=basin_id, create_new_files=True):
                with open(file_name, newline="\n") as csvfile:
                    shape_file_reader = csv.reader(csvfile, delimiter=",")
                    shape_file_rows_list = list(shape_file_reader)
                    width = int(shape_file_rows_list[1][WIDTH_LOC_IN_ROW])
                    height = int(shape_file_rows_list[1][HEIGHT_LOC_IN_ROW])
                    if width > max_width:
                        max_width = width
                        basin_id_with_maximum_width = basin_id
                    if height > max_height:
                        max_height = height
                        basin_id_with_maximum_height = basin_id
        print(f"max width is: {max_width}")
        print(f"max height is: {max_height}")
        return int(max_width), int(max_height), basin_id_with_maximum_width, basin_id_with_maximum_height

    def read_all_dynamic_attributes(self, all_stations_ids, specific_model_type, max_width, max_height,
                                    create_new_files):
        raise NotImplementedError

    def read_all_static_attributes(self, limit_size_above_1000=False):
        raise NotImplementedError