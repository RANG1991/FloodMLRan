import os
from pathlib import Path
import re
import pandas as pd
import netCDF4 as nc
import xarray as xr
import numpy as np


def calc_intersection_station_ids_files(file_1, file_2):
    with open(file_1) as f1:
        station_ids_files_1 = set([station_id.strip() for station_id in f1.readlines()])
    with open(file_2) as f2:
        station_ids_files_2 = set([station_id.strip() for station_id in f2.readlines()])
    print(f"intersection: {sorted(station_ids_files_1.intersection(station_ids_files_2))}")
    print(f"difference 1 -> 2: {sorted(station_ids_files_1.difference(station_ids_files_2))}")
    print(f"difference 2 -> 1: {sorted(station_ids_files_2.difference(station_ids_files_1))}")
    diff_stations = [int(station_id) for station_id in sorted(station_ids_files_2.difference(station_ids_files_1))]
    df = pd.read_csv(r"../data/CAMELS_US/camels_attributes_v2.0/attributes_combined.csv").set_index("gauge_id")
    df.loc[diff_stations, "area_gages2"].to_csv("sizes_small_basins.csv")


def open_nc_radar():
    dataset = nc.Dataset("/sci/labs/efratmorin/ranga/FloodMLRan/data/stage4_nc_files/200201/ST4.2002010100.01h.nc")


def rename_checkpoint_files(checkpoint_files_folder):
    all_checkpoint_files = Path(checkpoint_files_folder).glob("*.*")
    for checkpoint_file in all_checkpoint_files:
        new_checkpoint_file_name = checkpoint_file.name.replace("TWO_LSTM_CNN_LSTM", "CNN_LSTM").replace("None",
                                                                                                         "all_basins")
        os.rename(checkpoint_file, checkpoint_file.parent / new_checkpoint_file_name)


def fix_wrong_aligned_images_files_radar():
    spatial_files = Path("/sci/labs/efratmorin/ranga/FloodMLRan/data/CAMELS_US/radar_all_data/").rglob(
        "precip24_spatial_*.nc")
    shape_files = Path("/sci/labs/efratmorin/ranga/FloodMLRan/data/CAMELS_US/radar_all_data/").rglob("shape_*.csv")
    info_files = Path("/sci/labs/efratmorin/ranga/FloodMLRan/data/CAMELS_US/radar_all_data/").rglob("info_*.txt")
    for station_data_file_spatial, shape_file, info_file in zip(spatial_files, shape_files, info_files):
        # fix spatial file
        ds = nc.Dataset(station_data_file_spatial)
        ds = xr.open_dataset(xr.backends.NetCDF4DataStore(ds))
        X_data_spatial = np.asarray(ds["precipitation"])
        X_data_spatial = np.flip(np.rot90(X_data_spatial, axes=(2,)), axis=(2,))
        ds["precipitation"] = X_data_spatial
        ds.to_netcdf(station_data_file_spatial)
        # fix shape file
        df_shape = pd.read_csv(shape_file)
        width = df_shape.iloc[1, 2]
        height = df_shape.iloc[1, 3]
        df_shape.iloc[1, 2] = height
        df_shape.iloc[1, 3] = width
        df_shape.to_csv(shape_file)


def main():
    fix_wrong_aligned_images_files_radar()


if __name__ == "__main__":
    main()
