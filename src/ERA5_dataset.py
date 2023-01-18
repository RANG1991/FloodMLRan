from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from glob import glob
from datetime import datetime
import matplotlib
import os
import math

matplotlib.use("AGG")

import netCDF4 as nc
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Pool
import csv
import xarray as xr

ATTRIBUTES_TO_TEXT_DESC = {
    "p_mean": "Mean daily precipitation",
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
    "surface_net_solar_radiation_mean": "surface_net_solar_radiation_mean",
}

STATIC_ATTRIBUTES_NAMES = [
    "ele_mt_sav",
    "slp_dg_sav",
    "basin_area",
    "for_pc_sse",
    "cly_pc_sav",
    "slt_pc_sav",
    "snd_pc_sav",
    "soc_th_sav",
    "p_mean",
    "pet_mean",
    "aridity",
    "frac_snow",
    "high_prec_freq",
    "high_prec_dur",
    "low_prec_freq",
    "low_prec_dur",
]

DYNAMIC_ATTRIBUTES_NAMES_CARAVAN = [
    "total_precipitation_sum",
    "temperature_2m_min",
    "temperature_2m_max",
    "potential_evaporation_sum",
    "surface_net_solar_radiation_mean",
]

DISCHARGE_STR_CARAVAN = "streamflow"

DYNAMIC_DATA_FOLDER_CARAVAN = "../data/ERA5/Caravan/timeseries/csv/us/"

DISCHARGE_DATA_FOLDER_CARAVAN = "../data/ERA5/Caravan/timeseries/csv/us/"

DYNAMIC_ATTRIBUTES_NAMES_ERA5 = ["precip"]

DISCHARGE_STR_ERA5 = "flow"

DYNAMIC_DATA_FOLDER_ERA5 = "../data/ERA5/ERA_5_all_data"

DISCHARGE_DATA_FOLDER_ERA5 = "../data/ERA5/ERA_5_all_data"

STATIC_DATA_FOLDER = "../data/ERA5/Caravan/attributes"


class Dataset_ERA5(Dataset):
    def __init__(
            self,
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
            specific_model_type="",
            static_attributes_names=[],
            sequence_length=270,
            x_means=None,
            x_stds=None,
            y_mean=None,
            y_std=None,
            use_Caravan_dataset=True,

    ):
        self.x_mean_dict = {}
        self.x_std_dict = {}
        self.x_means = x_means if x_means is not None else None
        self.x_stds = x_stds if x_stds is not None else None
        self.y_mean_dict = {}
        self.y_std_dict = {}
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
        self.df_attr, self.list_stations_static = self.read_static_attributes()
        self.use_Caravan_dataset = use_Caravan_dataset
        self.specific_model_type = specific_model_type
        self.prefix_dynamic_data_file = "us_" if use_Caravan_dataset else "data24_"
        max_width, max_length = self.get_maximum_width_and_length_of_basin(
            "../data/ERA5/ERA_5_all_data"
        )
        if max_length <= 0 or max_width <= 0:
            raise Exception("max length or max width are not greater than 0")
        self.max_width = max(max_width, max_length)
        self.max_length = max(max_width, max_length)
        (self.dict_station_id_to_data,
         x_means,
         x_stds,
         y_mean,
         y_std
         ) = self.read_all_dynamic_data_files(all_stations_ids=all_stations_ids,
                                              specific_model_type=specific_model_type,
                                              max_width=self.max_width, max_length=self.max_length)

        self.y_mean = y_mean if stage == "train" else self.y_mean
        self.y_std = y_std if stage == "train" else self.y_std
        self.x_means = x_means if stage == "train" else self.x_means
        self.x_stds = x_stds if stage == "train" else self.x_stds

        self.dataset_length, self.lookup_table = self.create_look_table()
        x_data_mean_dynamic = self.x_means[:(len(self.list_dynamic_attributes_names))]
        x_data_std_dynamic = self.x_stds[:(len(self.list_dynamic_attributes_names))]

        x_data_mean_static = self.df_attr[self.list_static_attributes_names].mean().to_numpy()
        x_data_std_static = self.df_attr[self.list_static_attributes_names].std().to_numpy()

        for key in self.dict_station_id_to_data.keys():
            current_x_data = self.dict_station_id_to_data[key][0]
            current_y_data = self.dict_station_id_to_data[key][1]

            indices_features_dynamic_non_spatial = range(0, (len(self.list_dynamic_attributes_names)))

            current_x_data[:, indices_features_dynamic_non_spatial] = \
                (current_x_data[:, indices_features_dynamic_non_spatial] - x_data_mean_dynamic) / \
                (x_data_std_dynamic + (10 ** (-6)))

            indices_features_static = range((len(self.list_dynamic_attributes_names)),
                                            (len(self.list_dynamic_attributes_names))
                                            + (len(self.list_static_attributes_names)))

            current_x_data[:, indices_features_static] = \
                (current_x_data[:, indices_features_static] - x_data_mean_static) / (x_data_std_static + (10 ** (-6)))

            current_y_data = (current_y_data - self.y_mean) / (self.y_std + (10 ** (-6)))

            if specific_model_type.lower() == "lstm":
                self.dict_station_id_to_data[key] = (current_x_data, current_y_data)
            else:
                current_x_data_spatial = current_x_data[:, ((len(self.list_dynamic_attributes_names))
                                                            + (len(self.list_static_attributes_names))):].reshape(-1,
                                                                                                                  self.max_width,
                                                                                                                  self.max_length)
                current_x_data_spatial = ((current_x_data_spatial - self.x_means[-1])
                                          / (self.x_stds[-1] + (10 ** (-6))))
                indices_all_features_non_spatial = range(0,
                                                         (len(self.list_dynamic_attributes_names))
                                                         + (len(self.list_static_attributes_names)))
                current_x_data_non_spatial = current_x_data[:, indices_all_features_non_spatial]
                self.dict_station_id_to_data[key] = (current_x_data_non_spatial, current_x_data_spatial, current_y_data)

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

    def __getitem__(self, index) -> T_co:
        next_basin = self.lookup_table[index]
        if self.current_basin != next_basin:
            self.current_basin = next_basin
            self.inner_index_in_data_of_basin = 0
        if self.specific_model_type.lower() == "lstm":
            X_data, y_data = self.dict_station_id_to_data[self.current_basin]
        else:
            X_data, X_data_spatial, y_data = self.dict_station_id_to_data[self.current_basin]
            X_data_spatial = X_data_spatial.reshape(-1, self.max_width * self.max_length)
            X_data = np.concatenate([X_data, X_data_spatial], axis=1)
        X_data_tensor = torch.tensor(
            X_data[self.inner_index_in_data_of_basin: self.inner_index_in_data_of_basin + self.sequence_length]
        ).to(torch.float32)
        y_data_tensor = torch.tensor(
            y_data[self.inner_index_in_data_of_basin: self.inner_index_in_data_of_basin + self.sequence_length]
        ).to(torch.float32).squeeze()
        self.inner_index_in_data_of_basin += 1
        return self.current_basin, X_data_tensor, y_data_tensor

    def read_static_attributes(self):
        df_attr_caravan = pd.read_csv(
            Path(self.static_data_folder) / "attributes_hydroatlas_us.csv",
            dtype={"gauge_id": str},
        )
        df_attr_hydroatlas = pd.read_csv(
            Path(self.static_data_folder) / "attributes_caravan_us.csv",
            dtype={"gauge_id": str},
        )
        df_attr = df_attr_caravan.merge(df_attr_hydroatlas, on="gauge_id")
        df_attr["gauge_id"] = (
            df_attr["gauge_id"]
            .apply(lambda x: str(x).replace("us_", ""))
            .values.tolist()
        )
        df_attr = df_attr.dropna()
        df_attr = df_attr[["gauge_id"] + self.list_static_attributes_names]
        # maxes = df_attr.drop(columns=['gauge_id']).max(axis=1).to_numpy().reshape(-1, 1)
        # mins = df_attr.drop(columns=['gauge_id']).min(axis=1).to_numpy().reshape(-1, 1)
        df_attr[self.list_static_attributes_names] = df_attr.drop(
            columns=["gauge_id"]
        ).to_numpy()
        return df_attr, df_attr["gauge_id"].values.tolist()

    def read_all_dynamic_data_files(self, all_stations_ids, specific_model_type, max_width, max_length):
        cumm_m_x = 0
        cumm_s_x = 0
        cumm_m_x_spatial = 0
        cumm_s_x_spatial = 0
        cumm_m_y = 0
        cumm_s_y = 0
        count_of_samples = 0
        dict_station_id_to_data = {}
        for station_id in all_stations_ids:
            if self.check_is_valid_station_id(station_id):
                if (specific_model_type.lower() == "conv" or
                        specific_model_type.lower() == "cnn" or
                        specific_model_type.lower() == "transformer"):
                    X_data_spatial, y_data = self.read_single_station_file_spatial(station_id)
                    X_data, _ = self.read_single_station_file(station_id)
                    if len(X_data_spatial) == 0 or len(y_data) == 0 or len(X_data) == 0:
                        continue
                    X_data_spatial = self.crop_or_pad_precip_spatial(X_data_spatial, max_width, max_length)
                    if X_data.shape[0] != X_data_spatial.shape[0]:
                        print(f"spatial data does not aligned with non spatial data in basin: {station_id}")
                        continue
                else:
                    X_data, y_data = self.read_single_station_file(station_id)
                    if len(X_data) == 0 or len(y_data) == 0:
                        continue

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

                if (specific_model_type.lower() == "conv" or
                        specific_model_type.lower() == "cnn" or
                        specific_model_type.lower() == "transformer"):
                    prev_mean_x_spatial = cumm_m_x_spatial
                    count_of_samples = count_of_samples + (len(y_data))
                    cumm_m_x_spatial = cumm_m_x_spatial + ((X_data_spatial.mean((-2, -1)) - cumm_m_x_spatial)
                                                           / count_of_samples)
                    cumm_s_x_spatial = cumm_s_x_spatial + ((X_data_spatial.std((-2, -1)) - cumm_m_x_spatial)
                                                           * (X_data_spatial - prev_mean_x_spatial))
                if (specific_model_type.lower() == "conv" or
                        specific_model_type.lower() == "cnn" or
                        specific_model_type.lower() == "transformer"):
                    X_data = np.concatenate([X_data, X_data_spatial.reshape(X_data.shape[0],
                                                                            max_length * max_width)], axis=1)
                    cumm_m_x = np.concatenate([cumm_m_x, cumm_m_x_spatial])
                    cumm_s_x = np.concatenate([cumm_s_x, cumm_s_x_spatial])
                dict_station_id_to_data[station_id] = (X_data, y_data)

            else:
                print(f"station with id: {station_id} has no valid file")
        std_x = np.sqrt(cumm_s_x / (count_of_samples - 1))
        std_y = np.sqrt(cumm_s_y / (count_of_samples - 1))
        return dict_station_id_to_data, cumm_m_x, std_x, cumm_m_y.item(), std_y.item()

    def check_is_valid_station_id(self, station_id):
        return (station_id in self.list_stations_static
                and os.path.exists(Path(self.dynamic_data_folder) / f"{self.prefix_dynamic_data_file}{station_id}.csv")
                and os.path.exists(Path(DYNAMIC_DATA_FOLDER_ERA5) / f"precip24_spatial_{station_id}.nc"))

    def read_single_station_file_spatial(self, station_id):
        station_data_file_spatial = (
                Path(DYNAMIC_DATA_FOLDER_ERA5) / f"precip24_spatial_{station_id}.nc"
        )
        station_data_file_discharge = (
                Path(self.dynamic_data_folder)
                / f"{self.prefix_dynamic_data_file}{station_id}.csv"
        )
        ds = nc.Dataset(station_data_file_spatial)
        ds = xr.open_dataset(xr.backends.NetCDF4DataStore(ds))
        df_dis_data = pd.read_csv(station_data_file_discharge)
        (
            dataset_xarray_filtered,
            df_dis_data_filtered,
        ) = self.read_and_filter_dynamic_data_spatial(ds, df_dis_data)
        X_data_spatial = np.asarray(dataset_xarray_filtered["precipitation"])
        y_data = df_dis_data_filtered[self.discharge_str].to_numpy().flatten()
        y_data = y_data.reshape(-1, 1)
        static_attrib_station = (
            (self.df_attr[self.df_attr["gauge_id"] == station_id])
            .drop("gauge_id", axis=1)
            .to_numpy()
            .reshape(1, -1)
        )
        if self.stage == "train":
            if station_id not in self.y_mean_dict.keys():
                self.y_mean_dict[station_id] = torch.tensor(y_data.mean(axis=0))
            if station_id not in self.y_std_dict.keys():
                self.y_std_dict[station_id] = torch.tensor(y_data.std(axis=0))
        return X_data_spatial, y_data

    def read_and_filter_dynamic_data_spatial(self, dataset_xarray, df_dis_data):
        df_dis_data.loc[self.discharge_str] = df_dis_data[self.discharge_str].apply(
            lambda x: float(x)
        )
        df_dis_data = df_dis_data[df_dis_data[self.discharge_str] >= 0]
        df_dis_data = df_dis_data.dropna()
        df_dis_data["date"] = pd.to_datetime(df_dis_data.date)
        start_date = (
            self.train_start_date
            if self.stage == "train"
            else self.test_start_date
            if self.stage == "test"
            else self.validation_start_date
        )
        start_date = datetime.strptime(start_date, "%d/%m/%Y")
        end_date = (
            self.train_end_date
            if self.stage == "train"
            else self.test_end_date
            if self.stage == "test"
            else self.validation_end_date
        )
        end_date = datetime.strptime(end_date, "%d/%m/%Y")
        df_dis_data = df_dis_data[
            (df_dis_data["date"] >= start_date) & (df_dis_data["date"] <= end_date)
            ]
        dataset_xarray["datetime"] = pd.DatetimeIndex(dataset_xarray["datetime"].values)
        dataset_xarray_filtered = dataset_xarray.isel(
            datetime=dataset_xarray.datetime.isin(df_dis_data["date"])
        )
        return dataset_xarray_filtered, df_dis_data

    def read_single_station_file(self, station_id):
        station_data_file = (
                Path(self.dynamic_data_folder)
                / f"{self.prefix_dynamic_data_file}{station_id}.csv"
        )
        df_dynamic_data = pd.read_csv(station_data_file)
        df_dynamic_data = self.read_and_filter_dynamic_data(df_dynamic_data)
        y_data = df_dynamic_data[self.discharge_str].to_numpy().flatten()
        X_data = df_dynamic_data[self.list_dynamic_attributes_names].to_numpy()
        if X_data.size == 0 or y_data.size == 0:
            return np.array([]), np.array([])
        X_data = X_data.reshape(-1, len(self.list_dynamic_attributes_names))
        y_data = y_data.reshape(-1, 1)
        static_attrib_station = (
            (self.df_attr[self.df_attr["gauge_id"] == station_id])
            .drop("gauge_id", axis=1)
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
        station_id_repeated = [station_id] * X_data.shape[0]
        if self.stage == "train":
            if station_id not in self.y_mean_dict.keys():
                self.y_mean_dict[station_id] = torch.tensor(y_data.mean(axis=0))
            if station_id not in self.y_std_dict.keys():
                self.y_std_dict[station_id] = torch.tensor(y_data.std(axis=0))
        # y_data -= (self.y_mean_dict[station_id].numpy())
        # y_data /= (self.y_std_dict[station_id].numpy())
        return X_data, y_data

    def read_and_filter_dynamic_data(self, df_dynamic_data):
        df_dynamic_data = df_dynamic_data[
            self.list_dynamic_attributes_names + [self.discharge_str] + ["date"]
            ].copy()
        df_dynamic_data[self.list_dynamic_attributes_names] = df_dynamic_data[
            self.list_dynamic_attributes_names
        ].astype(float)
        df_dynamic_data.loc[self.discharge_str] = df_dynamic_data[
            self.discharge_str
        ].apply(lambda x: float(x))
        df_dynamic_data = df_dynamic_data[df_dynamic_data[self.discharge_str] >= 0]
        dynamic_attributes_to_get_from_df = self.list_dynamic_attributes_names[0] if len(
            self.list_dynamic_attributes_names) == 1 else self.list_dynamic_attributes_names
        # df_dynamic_data = df_dynamic_data[df_dynamic_data[dynamic_attributes_to_get_from_df] >= 0]
        df_dynamic_data = df_dynamic_data.dropna()
        df_dynamic_data["date"] = pd.to_datetime(df_dynamic_data.date)
        start_date = (
            self.train_start_date
            if self.stage == "train"
            else self.test_start_date
            if self.stage == "test"
            else self.validation_start_date
        )
        start_date = datetime.strptime(start_date, "%d/%m/%Y")
        end_date = (
            self.train_end_date
            if self.stage == "train"
            else self.test_end_date
            if self.stage == "test"
            else self.validation_end_date
        )
        end_date = datetime.strptime(end_date, "%d/%m/%Y")
        df_dynamic_data = df_dynamic_data[
            (df_dynamic_data["date"] >= start_date)
            & (df_dynamic_data["date"] <= end_date)
            ]
        return df_dynamic_data

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

    def calculate_dataset_length(self):
        return self.dataset_length

    def create_boxplot_of_entire_dataset(self):
        all_attributes_names = (
                self.list_static_attributes_names + self.list_dynamic_attributes_names
        )
        dict_boxplots_data = {}
        for i in range(self.X_data.shape[1]):
            dict_boxplots_data[
                ATTRIBUTES_TO_TEXT_DESC[all_attributes_names[i]]
            ] = self.X_data[:, i]
        Dataset_ERA5.create_boxplot_on_data(
            dict_boxplots_data, plot_title=f"{self.stage}"
        )

    @staticmethod
    def create_boxplot_on_data(dict_boxplots_data, plot_title=""):
        fig, ax = plt.subplots()
        dict_boxplots_data = {
            k: v
            for k, v in sorted(dict_boxplots_data.items(), key=lambda item: item[0])
        }
        box_plots = ax.boxplot(
            dict_boxplots_data.values(),
            showfliers=False,
            positions=list(range(1, len(dict_boxplots_data.keys()) + 1)),
        )
        ax.set_xticks(np.arange(1, len(dict_boxplots_data.keys()) + 1))
        plt.yticks(fontsize=6)
        plt.legend(
            [box_plots["boxes"][i] for i in range(len(dict_boxplots_data.keys()))],
            [f"{i + 1} :" + k for i, k in enumerate(dict_boxplots_data.keys())],
            fontsize=6,
            handlelength=0,
            handletextpad=0,
        )
        fig.tight_layout()
        curr_datetime = datetime.now()
        curr_datetime_str = curr_datetime.strftime("%d-%m-%Y_%H:%M:%S")
        plt.grid()
        plt.title(f"Box plots data - {plot_title}", fontsize=8)
        plt.savefig(
            f"../data/images/data_box_plots_{plot_title}"
            + f"_{curr_datetime_str}"
            + ".png"
        )

    @staticmethod
    def get_maximum_width_and_length_of_basin(shape_files_folder):
        WIDTH_LOC_IN_ROW = 2
        HEIGHT_LOC_IN_ROW = 3
        max_height = -1
        max_width = -1
        file_names = glob(f"{shape_files_folder}/shape_*.csv")
        for file_name in file_names:
            with open(file_name, newline="\n") as csvfile:
                shape_file_reader = csv.reader(csvfile, delimiter=",")
                shape_file_rows_list = list(shape_file_reader)
                width = int(shape_file_rows_list[1][WIDTH_LOC_IN_ROW])
                height = int(shape_file_rows_list[1][HEIGHT_LOC_IN_ROW])
                if width > max_width:
                    max_width = width
                if height > max_height:
                    max_height = height
        print(f"max width is: {max_width}")
        print(f"max height is: {max_height}")
        return int(max_width), int(max_height)

    def set_sequence_length(self, sequence_length):
        self.sequence_length = sequence_length
        self.dataset_length, self.lookup_table = self.create_look_table()

    def get_x_stds(self):
        return self.x_stds

    def get_x_means(self):
        return self.x_means

    def get_y_std(self):
        return self.y_std

    def get_y_mean(self):
        return self.y_mean
