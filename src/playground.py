import os
from pathlib import Path
import re
import pandas as pd
import netCDF4 as nc
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import CAMELS_dataset


def plot_training_and_validation_losses(slurm_output_file):
    file_name = slurm_output_file.name.replace('slurm-', '').replace('.out', '')
    training_loss_list = []
    validation_loss_list = []
    with open(slurm_output_file, "r", encoding="utf-8") as f:
        new_epoch_train = True
        new_epoch_validation = True
        for row in f:
            match_loss_training = re.search(r"Loss on the entire training epoch: (\d+\.\d+)", row)
            if match_loss_training and new_epoch_train:
                loss = float(match_loss_training.group(1))
                training_loss_list.append(loss)
                new_epoch_train = False
            match_loss_validation = re.search(r"Loss on the entire validation epoch: (\d+\.\d+)", row)
            if match_loss_validation and new_epoch_validation:
                loss = float(match_loss_validation.group(1))
                validation_loss_list.append(loss)
                new_epoch_validation = False
            if re.search("finished calculating the NSE per basin", row):
                new_epoch_train = True
                new_epoch_validation = True
    plt.title(f"training_and_validation_loss_of_{file_name}")
    plt.plot(training_loss_list, label="training")
    plt.plot(validation_loss_list, label="validation")
    plt.legend(loc="upper left")
    plt.savefig(f"training_and_validation_loss_of_{file_name}")
    plt.close()


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


def count_number_of_pixels_images():
    spatial_files = sorted(Path(
        f"/sci/labs/efratmorin/ranga/FloodMLRan/data/CAMELS_US/CAMELS_all_data/").rglob("precip24_spatial_*.nc"))
    max_num_non_zero_pixels = -1
    for station_data_file_spatial in spatial_files:
        ds_ncf = nc.Dataset(station_data_file_spatial, 'r')
        X_data_spatial = np.asarray(ds_ncf["precipitation"])
        non_zero_pixels = np.count_nonzero(X_data_spatial.sum(axis=0))
        if non_zero_pixels > max_num_non_zero_pixels:
            max_num_non_zero_pixels = non_zero_pixels
    print(max_num_non_zero_pixels)


def fix_wrong_aligned_images_files_CAMELS(basin_id=-1):
    if basin_id != -1:
        spatial_files = [Path(
            f"/sci/labs/efratmorin/ranga/FloodMLRan/data/CAMELS_US/CAMELS_all_data/precip24_spatial_{basin_id}.nc")]
    else:
        spatial_files = sorted(Path(
            f"/sci/labs/efratmorin/ranga/FloodMLRan/data/CAMELS_US/CAMELS_all_data/").rglob("precip24_spatial_*.nc"))
    for station_data_file_spatial in spatial_files:
        print(f"processing: {station_data_file_spatial.name}")
        try:
            # fix spatial file
            ds_ncf = nc.Dataset(station_data_file_spatial, 'r+')
            # ds = xr.open_dataset(xr.backends.NetCDF4DataStore(ds_ncf))
            X_data_spatial = np.asarray(ds_ncf["precipitation"])
            X_data_spatial = np.fliplr(X_data_spatial)
            # plt.imsave("check.png", X_data_spatial.sum(axis=0))
            ds_ncf["precipitation"][:] = X_data_spatial
            ds_ncf.close()
        except Exception as e:
            print(e)


def fix_wrong_aligned_images_files_radar(basin_id=-1):
    if basin_id != -1:
        spatial_files = Path(
            f"/sci/labs/efratmorin/ranga/FloodMLRan/data/CAMELS_US/radar_all_data/precip24_spatial_{basin_id}.nc")
        shape_files = Path(f"/sci/labs/efratmorin/ranga/FloodMLRan/data/CAMELS_US/radar_all_data/shape_{basin_id}.csv")
        info_files = Path(f"/sci/labs/efratmorin/ranga/FloodMLRan/data/CAMELS_US/radar_all_data/info_{basin_id}.txt")
    else:
        spatial_files = sorted(Path(
            f"/sci/labs/efratmorin/ranga/FloodMLRan/data/CAMELS_US/radar_all_data/").rglob("precip24_spatial_*.nc"))
        shape_files = sorted(
            Path("/sci/labs/efratmorin/ranga/FloodMLRan/data/CAMELS_US/radar_all_data/").rglob("shape_*.csv"))
        info_files = sorted(
            Path("/sci/labs/efratmorin/ranga/FloodMLRan/data/CAMELS_US/radar_all_data/").rglob("info_*.txt"))
    for station_data_file_spatial, shape_file, info_file in zip(spatial_files, shape_files, info_files):
        # fix spatial file
        ds_ncf = nc.Dataset(station_data_file_spatial, 'r+')
        # ds = xr.open_dataset(xr.backends.NetCDF4DataStore(ds_ncf))
        X_data_spatial = np.asarray(ds_ncf["precipitation"])
        X_data_spatial = X_data_spatial[:, ::-1, ::-1]
        # X_data_spatial = 0
        # plt.imsave("check.png", X_data_spatial.sum(axis=0))
        ds_ncf["precipitation"][:] = X_data_spatial
        ds_ncf.close()
        # fix shape file
        # df_shape = pd.read_csv(shape_file)
        # df_shape = df_shape[["time", "lat", "lon"]]
        # width = df_shape.iloc[0, 2]
        # height = df_shape.iloc[0, 3]
        # df_shape.iloc[0, 2] = height
        # df_shape.iloc[0, 3] = width
        # df_shape.to_csv(shape_file)
        # fix info file
        # with open(info_file) as f:
        #     f_text = f.read()
        #     f_text_split = f_text.split(" ")
        #     width = f_text_split[1]
        #     height = f_text_split[2]
        #     f_text_split[1] = height
        #     f_text_split[2] = width
        # with open(info_file, "w") as f:
        #     f.write(" ".join(f_text_split))


def check_stationary():
    camels_dataset = CAMELS_dataset.Dataset_CAMELS(
        main_folder=CAMELS_dataset.MAIN_FOLDER,
        dynamic_data_folder=CAMELS_dataset.DYNAMIC_DATA_FOLDER_NON_SPATIAL,
        static_data_folder=CAMELS_dataset.STATIC_DATA_FOLDER,
        dynamic_data_folder_spatial=CAMELS_dataset.DYNAMIC_DATA_FOLDER_SPATIAL_CAMELS,
        discharge_data_folder=CAMELS_dataset.DISCHARGE_DATA_FOLDER,
        dynamic_attributes_names=CAMELS_dataset.DYNAMIC_ATTRIBUTES_NAMES,
        static_attributes_names=CAMELS_dataset.STATIC_ATTRIBUTES_NAMES,
        train_start_date="01/10/1997",
        train_end_date="30/09/2002",
        validation_start_date="01/10/1988",
        validation_end_date="30/09/1992",
        test_start_date="01/10/1992",
        test_end_date="30/09/1997",
        stage="train",
        model_name="CNN_LSTM",
        sequence_length_spatial=14,
        create_new_files=False,
        all_stations_ids=sorted(open("../data/spatial_basins_list.txt").read().splitlines()),
        sequence_length=270,
        discharge_str=CAMELS_dataset.DISCHARGE_STR,
        use_all_static_attr=False,
        limit_size_above_1000=True,
        num_basins=None,
        use_only_precip_feature=False,
        run_with_radar_data=False
    )
    for i in range(10):
        _, _, xs_non_spatial, _, _, _ = camels_dataset[i]
        plt.plot(xs_non_spatial.numpy()[:, 0])
        plt.savefig("check.png")
        result = adfuller(xs_non_spatial.numpy()[:, 0])
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print('Critical Values:')
        for key, value in result[4].items():
            print('\t%s: %.3f' % (key, value))


def create_iowa_map():
    import pandas as pd
    import geopandas as gpd
    import plotly.graph_objects as go

    df_sample = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/minoritymajority.csv')
    df_sample_r = df_sample[df_sample['STNAME'] == 'Iowa']
    df_sample_r['FIPS'] = df_sample_r['FIPS'].astype('str')
    url = 'https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json'
    geo_counties = gpd.read_file(url)
    geo_df = geo_counties.merge(df_sample_r, left_on='id', right_on='FIPS')
    geo_df = geo_df.set_index('id')
    fig = go.Figure(go.Choroplethmapbox(geojson=geo_df.__geo_interface__,
                                        locations=geo_df.index,
                                        z=geo_df['TOT_POP'],
                                        colorscale="Viridis",
                                        marker_opacity=0.5,
                                        marker_line_width=0.5
                                        ))
    fig.update_layout(mapbox_style="carto-positron",
                      mapbox_zoom=5,
                      mapbox_center={"lat": 42.032974, "lon": -93.581543})
    fig.update_layout(height=600, margin={"r": 0, "t": 20, "l": 0, "b": 0})
    fig.write_image("iowa_map.png")


def main():
    plot_training_and_validation_losses(
        Path("../slurm_output_files/slurm_files_ensemble_comparison/slurm-19158233.out").resolve())


if __name__ == "__main__":
    main()
