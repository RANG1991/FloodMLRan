import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
import matplotlib
import os
from tqdm import tqdm
import sys
import psutil
import gc
import codecs
import json
from FloodML_Base_Dataset import FloodML_Base_Dataset

matplotlib.use("AGG")

import netCDF4 as nc
import matplotlib.pyplot as plt
import xarray as xr
import cv2

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
    "p_mean",
    "pet_mean",
    "aridity",
    "seasonality",
    "frac_snow",
    "high_prec_freq",
    "high_prec_dur",
    "low_prec_freq",
    "low_prec_dur",
    "ele_mt_sav",
    "slp_dg_sav",
    "basin_area",
    "for_pc_sse",
    "cly_pc_sav",
    "slt_pc_sav",
    "snd_pc_sav",
    "soc_th_sav",
]

DYNAMIC_ATTRIBUTES_NAMES_CARAVAN = [
    "total_precipitation_sum",
    "temperature_2m_min",
    "temperature_2m_max",
    "potential_evaporation_sum",
    "surface_net_solar_radiation_mean",
]

DISCHARGE_STR_CARAVAN = "streamflow"

DYNAMIC_DATA_FOLDER_CARAVAN = "../data/ERA5/Caravan/timeseries/csv/"

DISCHARGE_DATA_FOLDER_CARAVAN = "../data/ERA5/Caravan/timeseries/csv/"

DYNAMIC_ATTRIBUTES_NAMES_ERA5 = ["precip"]

DISCHARGE_STR_ERA5 = "flow"

DYNAMIC_DATA_FOLDER_ERA5 = "../data/ERA5/ERA_5_all_data"

DISCHARGE_DATA_FOLDER_ERA5 = "../data/ERA5/ERA_5_all_data"

STATIC_DATA_FOLDER = "../data/ERA5/Caravan/attributes"

MAIN_FOLDER = "../data/ERA5/"

RESIZE_WIDTH = 10
RESIZE_HEIGHT = 10


class Dataset_ERA5(FloodML_Base_Dataset):
    def __init__(
            self,
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
            use_Caravan_dataset=True,
            create_new_files=False,
            limit_size_above_1000=False,
            use_all_static_attr=False
    ):
        self.countries_abbreviations_stations_dict = {}
        self.countries_abbreviations = ["au", "br", "ca", "cl", "gb", "lamah", "us"]
        self.use_Caravan_dataset = use_Caravan_dataset
        super().__init__(
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
            specific_model_type,
            static_attributes_names,
            sequence_length,
            x_means,
            x_stds,
            y_mean,
            y_std,
            create_new_files,
            limit_size_above_1000,
            use_all_static_attr)

    def read_static_attributes_single_country(self, country_abbreviation, countries_abbreviations_stations_dict,
                                              limit_size_above_1000=False):
        df_attr_caravan = pd.read_csv(
            Path(self.static_data_folder) / country_abbreviation / f"attributes_hydroatlas_{country_abbreviation}.csv",
            dtype={"gauge_id": str},
        )
        df_attr_hydroatlas = pd.read_csv(
            Path(self.static_data_folder) / country_abbreviation / f"attributes_caravan_{country_abbreviation}.csv",
            dtype={"gauge_id": str},
        )
        df_attr = df_attr_caravan.merge(df_attr_hydroatlas, on="gauge_id")
        df_attr["gauge_id"] = (
            df_attr["gauge_id"]
            .apply(lambda x: str(x).replace(f"{country_abbreviation}_", ""))
            .values.tolist()
        )
        df_attr = df_attr.dropna()
        if limit_size_above_1000:
            df_attr = df_attr[df_attr["basin_area"] >= 1000]
        if self.use_all_static_attr:
            self.list_static_attributes_names = df_attr.columns.to_list()
            if "gauge_id" in self.list_static_attributes_names:
                self.list_static_attributes_names.remove("gauge_id")
        df_attr = df_attr[["gauge_id"] + self.list_static_attributes_names]
        # maxes = df_attr.drop(columns=['gauge_id']).max(axis=1).to_numpy().reshape(-1, 1)
        # mins = df_attr.drop(columns=['gauge_id']).min(axis=1).to_numpy().reshape(-1, 1)
        df_attr[self.list_static_attributes_names] = df_attr.drop(
            columns=["gauge_id"]
        ).to_numpy()
        list_station_ids = df_attr["gauge_id"].values.tolist()
        for station_id in list_station_ids:
            countries_abbreviations_stations_dict[station_id] = country_abbreviation
        return df_attr, list_station_ids

    def check_is_valid_station_id(self, station_id, create_new_files):
        if station_id not in self.countries_abbreviations_stations_dict.keys():
            return False
        country_abbreviation = self.countries_abbreviations_stations_dict[station_id]
        return (station_id in self.list_stations_static
                and os.path.exists(Path(f"{self.dynamic_data_folder}/{country_abbreviation}")
                                   / f"{country_abbreviation}_{station_id}.csv")
                and os.path.exists(Path(DYNAMIC_DATA_FOLDER_ERA5) / f"precip24_spatial_{station_id}.nc")
                and (not os.path.exists(
                    f"{self.folder_with_basins_pickles}/{station_id}_{self.stage}{self.suffix_pickle_file}.pkl")
                     or any([not os.path.exists(
                            f"{self.folder_with_basins_pickles}/mean_std_count_of_data.json_{self.stage}{self.suffix_pickle_file}"),
                             station_id not in self.x_mean_dict,
                             station_id not in self.x_std_dict,
                             station_id not in self.y_mean_dict,
                             station_id not in self.y_std_dict,
                             create_new_files])))

    def read_all_static_attributes(self, limit_size_above_1000=False):
        list_static_df = []
        for country_abbreviation in self.countries_abbreviations:
            curr_df, curr_list_stations = \
                self.read_static_attributes_single_country(country_abbreviation,
                                                           self.countries_abbreviations_stations_dict,
                                                           limit_size_above_1000=limit_size_above_1000)
            self.list_stations_static.extend(curr_list_stations)
            list_static_df.append(curr_df)
        df_attr = pd.concat(list_static_df)
        return df_attr, self.list_stations_static

    def read_all_dynamic_attributes(self, all_stations_ids, specific_model_type, max_width, max_height,
                                    create_new_files):
        if os.path.exists(
                f"{self.folder_with_basins_pickles}/mean_std_count_of_data.json_{self.stage}{self.suffix_pickle_file}") and not create_new_files:
            obj_text = codecs.open(
                f"{self.folder_with_basins_pickles}/mean_std_count_of_data.json_{self.stage}{self.suffix_pickle_file}",
                'r',
                encoding='utf-8').read()
            json_obj = json.loads(obj_text)
            cumm_m_x = np.array(json_obj["cumm_m_x"])
            cumm_s_x = np.array(json_obj["cumm_s_x"])
            cumm_m_x_spatial = np.array(json_obj["cumm_m_x_spatial"])
            cumm_s_x_spatial = np.array(json_obj["cumm_s_x_spatial"])
            cumm_m_y = float(json_obj["cumm_m_y"])
            cumm_s_y = float(json_obj["cumm_s_y"])
            count_of_samples = int(json_obj["count_of_samples"])
        else:
            cumm_m_x = 0
            cumm_s_x = 0
            cumm_m_x_spatial = -1
            cumm_s_x_spatial = -1
            cumm_m_y = 0
            cumm_s_y = 0
            count_of_samples = 0
        dict_station_id_to_data = {}
        pbar = tqdm(all_stations_ids, file=sys.stdout)
        pbar.set_description(f"processing basins - {self.stage}")
        for station_id in pbar:
            print('RAM Used (GB):', psutil.virtual_memory()[3] / 1000000000)
            if self.check_is_valid_station_id(station_id, create_new_files=create_new_files):
                if (specific_model_type.lower() == "conv" or
                        specific_model_type.lower() == "cnn"):
                    X_data_spatial, _ = self.read_single_station_file_spatial(station_id)
                    X_data_non_spatial, y_data = self.read_single_station_file(station_id)
                    if len(X_data_spatial) == 0 or len(y_data) == 0 or len(X_data_non_spatial) == 0:
                        del X_data_spatial
                        del X_data_non_spatial
                        del y_data
                        continue
                    max_dim = max(max_width, max_height)
                    X_data_spatial_list = []
                    for i in range(X_data_spatial.shape[0]):
                        try:
                            X_data_spatial_list.append(
                                np.expand_dims(cv2.resize(X_data_spatial[i, :, :].squeeze(), (max_dim, max_dim),
                                                          interpolation=cv2.INTER_LINEAR), axis=0))
                        except Exception:
                            X_data_spatial_list.append(
                                self.crop_or_pad_precip_spatial(X_data_spatial[i, :, :], max_dim, max_dim))
                    X_data_spatial = np.concatenate(X_data_spatial_list)
                    # X_data_spatial = self.crop_or_pad_precip_spatial(X_data_spatial, max_dim, max_dim)
                    gray_image = X_data_spatial.reshape((X_data_spatial.shape[0], max_dim, max_dim)).sum(axis=0)
                    plt.imsave(f"../data/basin_check_precip_images/img_{station_id}_precip.png",
                               gray_image)
                    if X_data_non_spatial.shape[0] != X_data_spatial.shape[0]:
                        print(f"spatial data does not aligned with non spatial data in basin: {station_id}")
                        del X_data_spatial
                        del X_data_non_spatial
                        del y_data
                        continue
                else:
                    X_data_non_spatial, y_data = self.read_single_station_file(station_id)
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

                if (specific_model_type.lower() == "conv" or
                        specific_model_type.lower() == "cnn"):
                    X_data_spatial = np.array(
                        X_data_spatial.reshape(X_data_non_spatial.shape[0], self.max_dim * self.max_dim),
                        dtype=np.float64)

                    prev_mean_x_spatial = cumm_m_x_spatial
                    cumm_m_x_spatial = cumm_m_x_spatial + (
                            (X_data_spatial[:] - cumm_m_x_spatial) / count_of_samples).sum(
                        axis=0)
                    cumm_s_x_spatial = cumm_s_x_spatial + (
                            (X_data_spatial[:] - cumm_m_x_spatial) * (X_data_spatial[:] - prev_mean_x_spatial)).sum(
                        axis=0)

                    X_data_non_spatial = np.concatenate([X_data_non_spatial, X_data_spatial], axis=1)
                    del X_data_spatial
                dict_station_id_to_data[station_id] = {"x_data": X_data_non_spatial, "y_data": y_data}

            else:
                print(f"station with id: {station_id} has no valid file or the file already exists")
        gc.collect()
        std_x = np.sqrt(cumm_s_x / (count_of_samples - 1))
        std_y = np.sqrt(cumm_s_y / (count_of_samples - 1)).item()
        std_x_spatial = np.sqrt(cumm_s_x_spatial / (count_of_samples - 1))
        with codecs.open(
                f"{self.folder_with_basins_pickles}/mean_std_count_of_data.json_{self.stage}{self.suffix_pickle_file}",
                'w',
                encoding='utf-8') as json_file:
            json_obj = {
                "cumm_m_x": cumm_m_x.tolist(),
                "cumm_s_x": cumm_s_x.tolist(),
                "cumm_m_y": cumm_m_y,
                "cumm_s_y": cumm_s_y,
                "count_of_samples": count_of_samples
            }
            if specific_model_type.lower() == "conv" or specific_model_type.lower() == "cnn":
                json_obj["cumm_m_x_spatial"] = cumm_m_x_spatial.tolist()
                json_obj["cumm_s_x_spatial"] = cumm_s_x_spatial.tolist()
            json.dump(json_obj, json_file, separators=(',', ':'), sort_keys=True, indent=4)
        return dict_station_id_to_data, cumm_m_x, std_x, cumm_m_y, std_y, cumm_m_x_spatial, std_x_spatial

    def read_single_station_file_spatial(self, station_id):
        country_abbreviation = self.countries_abbreviations_stations_dict[station_id]
        station_data_file_spatial = (
                Path(DYNAMIC_DATA_FOLDER_ERA5) / f"precip24_spatial_{station_id}.nc"
        )
        station_data_file_discharge = (
                Path(f"{self.dynamic_data_folder}/{country_abbreviation}")
                / f"{country_abbreviation}_{station_id}.csv"
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
        country_abbreviation = self.countries_abbreviations_stations_dict[station_id]
        station_data_file = (
                Path(f"{self.dynamic_data_folder}/{country_abbreviation}")
                / f"{country_abbreviation}_{station_id}.csv"
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
        curr_datetime_str = curr_datetime.strftime("%d-%m-%Y_%H_%M_%S")
        plt.grid()
        plt.title(f"Box plots data - {plot_title}", fontsize=8)
        plt.savefig(
            f"../data/images/data_box_plots_{plot_title}"
            + ".png"
        )
