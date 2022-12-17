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
            x_mins=None,
            x_maxs=None,
            y_mean=None,
            y_std=None,
            use_Caravan_dataset=True,
    ):
        self.sequence_length = sequence_length
        self.dynamic_data_folder = dynamic_data_folder
        self.static_data_folder = static_data_folder
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
        self.dict_basin_records_count = {}
        (
            list_stations_repeated,
            X_data_list,
            y_data_list,
        ) = self.read_all_dynamic_data_files(all_stations_ids=all_stations_ids, specific_model_type=specific_model_type)
        self.X_data = np.concatenate(X_data_list)
        self.y_data = np.concatenate(y_data_list)
        self.y_std = y_std if y_std is not None else self.y_data.std()
        self.y_mean = y_mean if y_mean is not None else self.y_data.mean()
        self.x_min = x_mins if x_mins is not None else self.X_data.min(axis=0)
        self.x_max = x_maxs if x_maxs is not None else self.X_data.max(axis=0)
        self.X_data = (self.X_data - self.x_min) / (
                (self.x_max - self.x_min) + 10 ** (-6)
        )
        self.y_data = (self.y_data - self.y_mean) / self.y_std
        self.list_stations_repeated = list_stations_repeated

    @staticmethod
    def pad_np_array_equally_from_sides(X_data_single_basin, max_width, max_height):
        max_width_right = int(max_width / 2)
        max_width_left = math.ceil(max_width / 2)
        max_height_right = int(max_height / 2)
        max_height_left = math.ceil(max_height / 2)
        return np.pad(
            X_data_single_basin,
            (
                (0, 0),
                (
                    max_width_right - int(X_data_single_basin.shape[1] / 2),
                    max_width_left - math.ceil(X_data_single_basin.shape[1] / 2),
                ),
                (
                    max_height_right - int(X_data_single_basin.shape[2] / 2),
                    max_height_left - math.ceil(X_data_single_basin.shape[2] / 2),
                ),
            ),
            "constant",
            constant_values=0,
        )

    def __len__(self):
        return self.calculate_dataset_length()

    def __getitem__(self, index) -> T_co:
        X_data_tensor = torch.tensor(
            self.X_data[index: index + self.sequence_length]
        ).to(torch.float32)
        y_data_tensor = torch.tensor(self.y_data[index + self.sequence_length]).to(
            torch.float32
        )
        station_id = self.list_stations_repeated[index + self.sequence_length]
        return station_id, X_data_tensor, y_data_tensor

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

    def read_all_dynamic_data_files(self, all_stations_ids, specific_model_type):
        list_stations_repeated = []
        X_data_list = []
        y_data_list = []
        list_returned = []
        # with Pool(multiprocessing.cpu_count() - 1) as p:
        #     list_returned = p.map(
        #         self.read_single_station_file_spatial, all_stations_ids
        #     )
        max_width, max_height = self.get_maximum_width_and_length_of_basin(
            "../data/ERA5/ERA_5_all_data"
        )
        for station_id in all_stations_ids:
            if self.check_is_valid_station_id(station_id):
                if specific_model_type.lower() == "conv":
                    station_id_repeated, X_data_spatial, y_data = self.read_single_station_file_spatial(station_id)
                    X_data_spatial = self.pad_np_array_equally_from_sides(
                        X_data_spatial, max_width, max_height
                    ).flatten()
                    list_returned.append((station_id_repeated, X_data_spatial, y_data))
                elif specific_model_type.lower() == "cnn":
                    station_id_repeated, X_data_spatial, y_data_spatial = self.read_single_station_file_spatial(
                        station_id)
                    _, X_data_non_spatial, _ = self.read_single_station_file(station_id)
                    if len(X_data_non_spatial) > 0 and len(X_data_non_spatial) > 0:
                        X_data_spatial = self.pad_np_array_equally_from_sides(
                            X_data_spatial, max_width, max_height
                        )
                        X_data_all = np.concatenate(
                            [X_data_spatial.reshape(len(X_data_non_spatial), max_height * max_width),
                             np.stack(X_data_non_spatial)], axis=1)
                        list_returned.append((station_id_repeated, X_data_all, y_data_spatial))
                else:
                    list_returned.append(self.read_single_station_file(station_id))
        for station_id_repeated, X_data_curr, y_data_curr in list_returned:
            if len(station_id_repeated) > 0:
                self.dict_basin_records_count[station_id_repeated[0]] = len(
                    station_id_repeated
                )
                list_stations_repeated.extend(station_id_repeated)
                if X_data_curr.size > 0:
                    X_data_list.append(X_data_curr)
                    y_data_list.append(y_data_curr)
        return list_stations_repeated, X_data_list, y_data_list

    def check_is_valid_station_id(self, station_id):
        return (station_id in self.list_stations_static
                and os.path.exists(Path(self.dynamic_data_folder) / f"precip24_spatial_{station_id}.nc")
                and os.path.exists(
                    Path(self.dynamic_data_folder) / f"{self.prefix_dynamic_data_file}{station_id}.csv"))

    def read_single_station_file_spatial(self, station_id):
        station_data_file_spatial = (
                Path(self.dynamic_data_folder) / f"precip24_spatial_{station_id}.nc"
        )
        station_data_file_discharge = (
                Path(self.dynamic_data_folder) / f"dis24_{station_id}.csv"
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
        station_id_repeated = [station_id] * X_data_spatial.shape[0]
        return station_id_repeated, X_data_spatial, y_data

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
            return np.array([]), np.array([]), np.array([])
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
        X_data = np.concatenate([X_data, static_attrib_station_rep], axis=1)
        station_id_repeated = [station_id] * X_data.shape[0]
        return station_id_repeated, X_data, y_data

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

    def calculate_dataset_length(self):
        count = 0
        for key in self.dict_basin_records_count.keys():
            count += self.dict_basin_records_count[key] - self.sequence_length
        return count

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
        self.calculate_dataset_length()

    def get_x_mins(self):
        return self.x_min

    def get_x_maxs(self):
        return self.x_max

    def get_y_std(self):
        return self.y_std

    def get_y_mean(self):
        return self.y_mean
