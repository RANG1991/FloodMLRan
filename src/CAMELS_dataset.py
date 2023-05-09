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
import os
from tqdm import tqdm
import sys
import netCDF4 as nc
import xarray as xr
from FloodML_Base_Dataset import FloodML_Base_Dataset
from matplotlib import pyplot as plt
import cv2

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

DYNAMIC_DATA_FOLDER_NON_SPATIAL = "../data/CAMELS_US/basin_mean_forcing"

STATIC_DATA_FOLDER = "../data/CAMELS_US/camels_attributes_v2.0"

DISCHARGE_DATA_FOLDER = "../data/CAMELS_US/usgs_streamflow"

DYNAMIC_DATA_FOLDER_NON_SPATIAL_AND_SPATIAL = "../data/CAMELS_US/CAMELS_all_data/"

MAIN_FOLDER = "../data/CAMELS_US/"


class Dataset_CAMELS(FloodML_Base_Dataset):
    def __init__(
            self,
            main_folder,
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
            sequence_length_spatial,
            model_name,
            create_new_files,
            static_attributes_names=[],
            sequence_length=270,
            x_means=None,
            x_stds=None,
            y_mean=None,
            y_std=None,
            limit_size_above_1000=False,
            use_all_static_attr=False,
            num_basins=None,
            min_spatial=None,
            max_spatial=None
    ):
        self.discharge_data_folder = discharge_data_folder
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
            model_name,
            static_attributes_names,
            sequence_length,
            x_means,
            x_stds,
            y_mean,
            y_std,
            create_new_files,
            limit_size_above_1000,
            use_all_static_attr,
            num_basins,
            min_spatial=min_spatial,
            max_spatial=max_spatial)

    def check_is_valid_station_id(self, station_id, create_new_files):
        return (station_id in self.all_station_ids
                and (not os.path.exists(
                    f"{self.folder_with_basins_pickles}/{station_id}_{self.stage}{self.suffix_pickle_file}.pkl")
                     or any([not os.path.exists(
                            f"{self.folder_with_basins_pickles}/mean_std_count_of_data.json_{self.stage}"),
                             station_id not in self.x_mean_dict,
                             station_id not in self.x_std_dict,
                             station_id not in self.y_mean_dict,
                             station_id not in self.y_std_dict,
                             create_new_files])))

    def get_basin_area(self, station_id):
        forcing_path = Path(self.dynamic_data_folder)
        file_path = list(forcing_path.glob(f"**/{station_id}_*_forcing_leap.txt"))
        file_path = file_path[0]
        with open(file_path, "r") as fp:
            # load area from header
            fp.readline()
            fp.readline()
            area = int(fp.readline())
        return area

    def read_discharge_data(self, station_id):
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
        # normalize discharge from cubic feet per second to mm per day
        basin_area = self.get_basin_area(station_id)
        df_discharge.QObs = (
                28316846.592 * df_discharge.QObs * 86400 / (basin_area * 10 ** 6)
        )
        df_discharge = df_discharge.drop(columns=["Year", "Mnth", "Day"])
        return df_discharge

    def read_single_station_file(self, station_id):
        if station_id not in self.all_station_ids:
            return np.array([]), np.array([]), np.array([])
        forcing_path = Path(self.dynamic_data_folder)
        file_path = list(forcing_path.glob(f"**/{station_id}_*_forcing_leap.txt"))
        file_path = file_path[0]
        with open(file_path, "r") as fp:
            # load area from header
            fp.readline()
            fp.readline()
            _ = int(fp.readline())
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
            df_forcing = df_forcing.drop(columns=["Year", "Mnth", "Day"])
            df_discharge = self.read_discharge_data(station_id)
            df_dynamic_data = df_forcing.merge(df_discharge, on="date")
            df_dynamic_data.columns = map(str.lower, df_dynamic_data.columns)
            df_dynamic_data, list_dates = self.read_and_filter_dynamic_data(df_dynamic_data)
            df_dynamic_data = df_dynamic_data.set_index("date")

            y_data = df_dynamic_data[self.discharge_str].to_numpy().flatten()
            X_data = df_dynamic_data[self.list_dynamic_attributes_names].to_numpy()
            if X_data.size == 0 or y_data.size == 0:
                return np.array([]), np.array([]), np.array([])
            X_data = X_data.reshape(-1, len(self.list_dynamic_attributes_names))
            y_data = y_data.reshape(-1, 1)
            df_only_selected_attrib = self.df_attr[self.list_static_attributes_names]
            static_attrib_station = (
                (df_only_selected_attrib[df_only_selected_attrib.index == station_id])
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
            return X_data, y_data, list_dates

    def read_and_filter_dynamic_data(self, df_dynamic_data):
        df_dynamic_data = df_dynamic_data[
            self.list_dynamic_attributes_names + [self.discharge_str, "date"]
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
            (df_dynamic_data["date"] >= start_date) & (df_dynamic_data["date"] <= end_date)
            ]
        list_dates = df_dynamic_data["date"].apply(lambda x: np.array([x.year, x.month, x.day])).tolist()
        return df_dynamic_data, list_dates

    def read_single_station_file_spatial(self, station_id):
        station_data_file_spatial = (
                Path(DYNAMIC_DATA_FOLDER_NON_SPATIAL_AND_SPATIAL) / f"precip24_spatial_{station_id}.nc")
        ds = nc.Dataset(station_data_file_spatial)
        ds = xr.open_dataset(xr.backends.NetCDF4DataStore(ds))
        df_dis_data = self.read_discharge_data(station_id)
        df_dis_data.columns = map(str.lower, df_dis_data.columns)
        (
            dataset_xarray_filtered,
            df_dis_data_filtered,
        ) = self.read_and_filter_dynamic_data_spatial(ds, df_dis_data)
        X_data_spatial = np.asarray(dataset_xarray_filtered["precipitation"])
        y_data = df_dis_data_filtered[self.discharge_str].to_numpy().flatten()
        y_data = y_data.reshape(-1, 1)
        static_attrib_station = (
            (self.df_attr[self.df_attr.index == station_id])
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
        # the index column is the date column
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

    def read_all_static_attributes(self, limit_size_above_1000=False):
        attributes_path = Path(self.static_data_folder)
        txt_files = attributes_path.glob("camels_*.txt")
        # Read-in attributes into one big dataframe
        dfs = []
        for txt_file in txt_files:
            df_temp = pd.read_csv(txt_file, sep=";", header=0, dtype={"gauge_id": str})
            df_temp = df_temp.set_index("gauge_id")
            dfs.append(df_temp)
        df = pd.concat(dfs, axis=1)
        if limit_size_above_1000:
            df = df[df["area_gages2"] >= 1000]
        # convert huc column to double-digit strings
        df["huc"] = df["huc_02"].apply(lambda x: str(x).zfill(2))
        df = df.drop("huc_02", axis=1)
        if self.use_all_static_attr:
            self.list_static_attributes_names = df.columns.to_list()
            for column_name in ["huc", "gauge_name", "gauge_lat", "gauge_lon", "geol_1st_class", "geol_2nd_class",
                                "high_prec_timing", "low_prec_timing", "dom_land_cover"]:
                if column_name in self.list_static_attributes_names:
                    self.list_static_attributes_names.remove(column_name)
        df = df[self.list_static_attributes_names]
        return df, df.index.tolist()

    def read_all_dynamic_attributes(self, all_stations_ids, model_name, max_width, max_height, create_new_files):
        if os.path.exists(
                f"{self.folder_with_basins_pickles}/mean_std_count_of_data.json_{self.stage}{self.suffix_pickle_file}") and not create_new_files:
            obj_text = codecs.open(
                f"{self.folder_with_basins_pickles}/mean_std_count_of_data.json_{self.stage}{self.suffix_pickle_file}",
                'r',
                encoding='utf-8').read()
            json_obj = json.loads(obj_text)
            cumm_m_x = np.array(json_obj["cumm_m_x"])
            cumm_s_x = np.array(json_obj["cumm_s_x"])
            if model_name.lower() == "conv_lstm" or model_name.lower() == "cnn_lstm" \
                    or model_name.lower() == "cnn_transformer":
                cumm_m_x_spatial = np.array(json_obj["cumm_m_x_spatial"])
                cumm_s_x_spatial = np.array(json_obj["cumm_s_x_spatial"])
                max_spatial = np.array(json_obj["max_spatial"])
                min_spatial = np.array(json_obj["min_spatial"])
            cumm_m_y = float(json_obj["cumm_m_y"])
            cumm_s_y = float(json_obj["cumm_s_y"])
            count_of_samples = int(json_obj["count_of_samples"])
        else:
            cumm_m_x = 0
            cumm_s_x = 0
            if model_name.lower() == "conv_lstm" or model_name.lower() == "cnn_lstm" \
                    or model_name.lower() == "cnn_transformer":
                cumm_m_x_spatial = -1
                cumm_s_x_spatial = -1
                max_spatial = -1
                min_spatial = -1
            cumm_m_y = 0
            cumm_s_y = 0
            count_of_samples = 0
        dict_station_id_to_data = {}
        pbar = tqdm(all_stations_ids, file=sys.stdout)
        pbar.set_description(f"processing basins - {self.stage}")
        num_exist_stations = 0
        num_not_in_list_stations = 0
        for station_id in pbar:
            # print('RAM Used (GB):', psutil.virtual_memory()[3] / 1000000000)
            if self.check_is_valid_station_id(station_id, create_new_files=create_new_files):
                if (model_name.lower() == "conv_lstm" or
                        model_name.lower() == "cnn_lstm" or
                        model_name.lower() == "cnn_transformer"):
                    if not os.path.exists(
                            f"{DYNAMIC_DATA_FOLDER_NON_SPATIAL_AND_SPATIAL}/precip24_spatial_{station_id}.nc"):
                        continue
                    X_data_spatial, _ = self.read_single_station_file_spatial(station_id)
                    X_data_non_spatial, y_data, list_dates = self.read_single_station_file(station_id)
                    if any([X_data_spatial.shape[1] == 0, X_data_spatial.shape[2] == 0]) or len(
                            y_data) == 0 or len(X_data_non_spatial) == 0 or np.count_nonzero(X_data_spatial) == 0:
                        print("some of the data is empty, deleting and skipping this basin")
                        del X_data_spatial
                        del X_data_non_spatial
                        del y_data
                        continue
                    X_data_spatial_list = []
                    for i in range(X_data_spatial.shape[0]):
                        X_data_spatial_list.append(
                            np.expand_dims(cv2.resize(X_data_spatial[i, :, :].squeeze(), (self.max_dim, self.max_dim),
                                                      interpolation=cv2.INTER_LINEAR), axis=0))
                    X_data_spatial = np.concatenate(X_data_spatial_list)
                    # X_data_spatial = self.crop_or_pad_precip_spatial(X_data_spatial, max_dim, max_dim)
                    gray_image = X_data_spatial.reshape(X_data_spatial.shape[0], self.max_dim, self.max_dim).sum(
                        axis=0)
                    plt.imsave(f"../data/basin_check_precip_images/img_{station_id}_precip.png",
                               gray_image)
                    if X_data_non_spatial.shape[0] != X_data_spatial.shape[0]:
                        print(f"spatial data does not aligned with non spatial data in basin: {station_id}")
                        del X_data_spatial
                        del X_data_non_spatial
                        del y_data
                        continue
                else:
                    X_data_non_spatial, y_data, list_dates = self.read_single_station_file(station_id)
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

                if (model_name.lower() == "conv_lstm" or
                        model_name.lower() == "cnn_lstm" or
                        model_name.lower() == "cnn_transformer"):
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

                    curr_max_spatial = np.max(X_data_spatial, axis=1, keep_dims=True)
                    curr_min_spatial = np.min(X_data_spatial, axis=1, keep_dims=True)
                    if type(max_spatial) == int and max_spatial == -1:
                        max_spatial = curr_max_spatial
                    else:
                        max_spatial = np.maximum(max_spatial, curr_max_spatial)
                    if type(min_spatial) == int and min_spatial == -1:
                        min_spatial = curr_min_spatial
                    else:
                        min_spatial = np.minimum(min_spatial, curr_min_spatial)

                    X_data_non_spatial = np.concatenate([X_data_non_spatial, X_data_spatial], axis=1)
                    del X_data_spatial
                dict_station_id_to_data[station_id] = {"x_data": X_data_non_spatial, "y_data": y_data,
                                                       "list_dates": list_dates}
            else:
                if station_id not in self.all_station_ids:
                    num_not_in_list_stations += 1
                else:
                    num_exist_stations += 1
        gc.collect()
        std_x = np.sqrt(cumm_s_x / (count_of_samples - 1))
        std_y = np.sqrt(cumm_s_y / (count_of_samples - 1)).item()
        if model_name.lower() == "conv_lstm" or model_name.lower() == "cnn_lstm" \
                or model_name.lower() == "cnn_transformer":
            std_x_spatial = np.sqrt(cumm_s_x_spatial / (count_of_samples - 1))
        else:
            std_x_spatial = None
            cumm_m_x_spatial = None
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
            if model_name.lower() == "conv_lstm" or model_name.lower() == "cnn_lstm" \
                    or model_name.lower() == "cnn_transformer":
                json_obj["cumm_m_x_spatial"] = cumm_m_x_spatial.tolist()
                json_obj["cumm_s_x_spatial"] = cumm_s_x_spatial.tolist()
            json.dump(json_obj, json_file, separators=(',', ':'), sort_keys=True, indent=4)
        print(f"went over {len(all_stations_ids)} stations from which {num_exist_stations} already "
              f"exists and {num_not_in_list_stations} not in the list")
        return (dict_station_id_to_data, cumm_m_x, std_x, cumm_m_y, std_y, cumm_m_x_spatial, std_x_spatial, min_spatial,
                max_spatial)
