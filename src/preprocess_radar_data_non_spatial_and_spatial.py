import netCDF4
import geopandas as gpd
import pandas as pd
import numpy as np
import datetime
from pathlib import Path
import xarray as xr
import concurrent.futures
import sys
from shapely.geometry import Point
from geopandas import GeoDataFrame
from matplotlib import pyplot as plt

np.set_printoptions(threshold=sys.maxsize)

PATH_ROOT = "../../FloodMLRan/data"
COUNTRY_ABBREVIATION = "us"
TEST_PERIOD = ("1989-10-01", "1999-09-30")
TRAINING_PERIOD = ("1999-10-01", "2008-09-30")

GRID_DELTA = 0.25


def check_if_discharge_file_exists(station_id, ERA5_discharge_data_folder_name):
    timezone_file = ERA5_discharge_data_folder_name + "/timezone_" + station_id + ".txt"
    discharge_file = ERA5_discharge_data_folder_name + "/dis_" + station_id + ".csv"
    list_files = [timezone_file, discharge_file]
    all_files_exist = all(Path(file_name).exists() for file_name in list_files)
    return all_files_exist


def check_if_all_precip_files_exist(station_id, output_folder_name):
    shape_file = output_folder_name + "/shape_" + station_id.replace(COUNTRY_ABBREVIATION, "") + ".csv"
    precip24_file = output_folder_name + "/precip24_spatial_" + station_id.replace(COUNTRY_ABBREVIATION, "") + ".nc"
    info_file = output_folder_name + "/info_" + station_id.replace(COUNTRY_ABBREVIATION, "") + ".txt"
    latlon_file = output_folder_name + "/latlon_" + station_id.replace(COUNTRY_ABBREVIATION, "") + ".csv"
    list_files = [shape_file, precip24_file, info_file, latlon_file]
    all_files_exist = all(Path(file_name).exists() for file_name in list_files)
    return all_files_exist


def get_utc_offset(longitude):
    # start_time = pd.to_datetime("1980-01-01 00:00:00")
    # end_time = pd.to_datetime("2021-09-15 00:00:00")
    # param_id = "00060"
    # data = InstantValueIO(
    #     start_date=start_time, end_date=end_time, station=station_id, parameter=param_id
    # )
    # datetimes = []
    # for series in data:
    #     datetimes = [r[0] for r in series.data]
    # utc_offset = datetimes[0].utcoffset().total_seconds() / 60 / 60
    utc_offset = round(longitude * 24 / 360)
    return utc_offset


# def parse_single_basin_discharge(station_id, basin_data, output_folder_name):
#     dis_file_exists = check_if_discharge_file_exists(station_id, output_folder_name)
#     if dis_file_exists:
#         print("The discharge file of the basin: {} exists".format(station_id))
#         return
#     start_time = pd.to_datetime("1980-01-01 00:00:00")
#     end_time = pd.to_datetime("2021-09-15 00:00:00")
#     param_id = "00060"
#     data = InstantValueIO(
#         start_date=start_time, end_date=end_time, station=station_id, parameter=param_id
#     )
#
#     FT2M = 0.3048
#     datetimes = []
#     flow = []
#     for series in data:
#         flow = [r[1] for r in series.data]
#         datetimes = [r[0] for r in series.data]
#
#     utc_offset = datetimes[0].utcoffset().total_seconds() / 60 / 60
#
#     area = basin_data["AREA"].values[0] / 1000000
#     ls = []
#     for i in range(0, len(datetimes)):
#         t = datetimes[i]
#         flow_curr = flow[i] * (FT2M ** 3) * 3.6 * 24 / area
#         ls.append([t.year, t.month, t.day, t.hour, t.minute, flow_curr])
#     df = pd.DataFrame(
#         data=ls, columns=["year", "month", "day", "hour", "minute", "flow"]
#     )
#     df_group = df.groupby(by=["year", "month", "day", "hour"]).mean()
#     df_group = df_group.assign(minute=0)
#     df_group.loc[df_group["flow"] < 0, "flow"] = 0
#
#     fn = output_folder_name + "/timezone_" + station_id + ".txt"
#     with open(fn, "w") as f:
#         print(utc_offset, file=f)
#     filename = output_folder_name + "/dis_" + station_id + ".csv"
#     df_group.to_csv(filename)


def get_index_by_lat_lon(lat, lon, lat_grid, lon_grid):
    i = np.where(lat == lat_grid)[0]
    assert len(i) > 0, f"Please supply latitude between {min(lat_grid)} and {max(lat_grid)}"
    j = np.where(lon == lon_grid)[0]
    assert len(j) > 0, f"Please supply longitude between {min(lon_grid)} and {max(lon_grid)}"
    return i, j


def create_and_write_precipitation_spatial(datetimes, ls_spatial, station_id, output_folder_name, mask_precip):
    ds = xr.Dataset(
        {
            "precipitation": xr.DataArray(
                data=ls_spatial,
                dims=["datetime", "lat", "lon"],
                coords={"datetime": datetimes},
            )
        },
        attrs={"creation_date": datetime.datetime.now()},
    )
    ds["precipitation"] = ds["precipitation"] * mask_precip
    ds = ds.resample(datetime="1D").sum()
    ds.to_netcdf(path=output_folder_name + "/precip24_spatial_" + station_id.replace(COUNTRY_ABBREVIATION, "") + ".nc")


def create_and_write_precipitation_mean(datetimes, ls, station_id, output_folder_name):
    # create a dataframe from the precipitation data and the timedates
    df_precip = pd.DataFrame(data=ls, index=datetimes, columns=["precip"])
    # downsample the precipitation data into 1D (1 day) bins and sum the values falling into the same bin
    df_precip_one_day = df_precip.resample("1D").sum()
    df_precip_one_day = df_precip_one_day.reset_index()
    df_precip_one_day = df_precip_one_day.rename(columns={"index": "date"})
    df_precip_one_day = df_precip_one_day[["date", "precip"]]
    print(df_precip_one_day)
    df_precip_one_day.to_csv(
        output_folder_name + "/precip24_" +
        station_id.replace(COUNTRY_ABBREVIATION, "") + ".csv", float_format="%6.1f")
    return df_precip_one_day


def get_longitude_and_latitude_points(lon_grid, lat_grid):
    lon_grid.reshape(-1, 1).flatten().tolist()
    lat_grid.reshape(-1, 1).flatten().tolist()
    points = np.vstack((lon_grid, lat_grid)).T
    return points, lon_grid.shape[0], lon_grid.shape[1]


def plot_lon_lat_on_world_map(lon_grid, lat_grid, file_name):
    d = {"Longitude": lon_grid.reshape(-1, 1).flatten().tolist(),
         "Latitude": lat_grid.reshape(-1, 1).flatten().tolist()}
    df = pd.DataFrame.from_dict(d)
    geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
    gdf = GeoDataFrame(df, geometry=geometry)
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    gdf.plot(ax=world.plot(figsize=(10, 6)), marker='o', color='red', markersize=15)
    plt.savefig(f"f{file_name}_plot_lat_lon.png")


def parse_single_basin_precipitation(
        station_id,
        basin_data,
        radar_precip_data_folder,
        output_folder_name
):
    # all_files_exist = check_if_all_precip_files_exist(station_id, output_folder_name)
    # if all_files_exist:
    #     print("all precipitation file of the basin: {} exists".format(station_id))
    #     return
    bounds = basin_data.bounds
    # get the minimum and maximum longitude and latitude (square boundaries)
    min_lon = np.squeeze(np.floor(bounds["minx"].values * 10) / 10)
    min_lat = np.squeeze(np.floor(bounds["miny"].values * 10) / 10)
    max_lon = np.squeeze(np.ceil(bounds["maxx"].values * 10) / 10)
    max_lat = np.squeeze(np.ceil(bounds["maxy"].values * 10) / 10)

    offset = get_utc_offset((min_lat + max_lat) / 2)
    list_of_dates_all_years = []
    list_of_total_precipitations_all_years = []
    started_reading_data = False
    for year in range(2002, 2020):
        print(f"parsing year: {year} of basin : {station_id}")
        all_datetimes_one_year = []
        all_tp_one_year = []
        for dataset_file in Path(f"{radar_precip_data_folder}").rglob(f"{year}*/*pr*.24h.nc"):
            dataset = netCDF4.Dataset(dataset_file)
            # ti is an array containing the dates as the number of hours since 1900-01-01 00:00
            # e.g. - [780168, 780169, 780170, ...]
            if not started_reading_data:
                lon_grid = dataset["longitude"][:]
                lat_grid = dataset["latitude"][:]
                lon_lat_points, height, width = get_longitude_and_latitude_points(lon_grid, lat_grid)
                plot_lon_lat_on_world_map(lon_grid, lat_grid, dataset_file.stem)
                grid = basin_data["geometry"].contains_points()
                mask_precip = grid.reshape(height, width)
                started_reading_data = True
            tp = np.asarray(dataset["tp"][:, mask_precip])
            # zero out any precipitation that is less than 0
            tp[tp < 0] = 0
            all_tp_one_year.append(tp)
            # the format of each file is - ST4.yyyymmddhh.xxh.Z
            datetime_str = dataset_file.name.split(".")[1][:-2]
            datetime_object = datetime.strptime(datetime_str, 'yyyymmdd')
            all_datetimes_one_year.append(datetime_object)

        # append the datetimes from specific year to the list of datetimes of all years
        list_of_dates_all_years.append(all_datetimes_one_year)
        # append the precipitations from specific year to the list of precipitations of all years
        list_of_total_precipitations_all_years.append(all_tp_one_year)

    # concatenate the datetimes from all the years
    datetimes = np.concatenate(list_of_dates_all_years, axis=0)
    # concatenate the precipitation data from all the years
    precip = np.concatenate(list_of_total_precipitations_all_years, axis=0)
    # take the mean of the precipitation data spatially (along the latitude and longitude)
    precip_mean_lat_lon = np.mean(precip, axis=(1, 2))
    # create a dataframe from the datetimes
    df_precip_times = pd.DataFrame(data=datetimes, index=datetimes)
    datetimes = df_precip_times.index.to_pydatetime()
    datetimes = [time + datetime.timedelta(hours=offset) for time in datetimes]
    ls = [[precip_mean_lat_lon[i]] for i in range(0, len(datetimes))]
    create_and_write_precipitation_mean(
        datetimes,
        ls,
        station_id,
        output_folder_name)

    lat_lon_lst = lon_lat_points.reshape(height, width)[mask_precip].flatten().tolist()
    fn = output_folder_name + "/latlon_" + station_id.replace(COUNTRY_ABBREVIATION, "") + ".csv"
    pd.DataFrame(data=lat_lon_lst, columns=["lat", "lon"]).to_csv(
        fn, index=False, float_format="%6.1f"
    )

    create_and_write_precipitation_spatial(
        datetimes,
        precip,
        station_id,
        output_folder_name,
        mask_precip
    )

    fn = output_folder_name + "/shape_" + station_id.replace(COUNTRY_ABBREVIATION, "") + ".csv"
    pd.DataFrame(
        data=np.array([(len(datetimes),), (precip.shape[1],), (precip.shape[2],)]).T,
        columns=["time", "lat", "lon"],
        index=[1],
    ).to_csv(fn)
    fn = output_folder_name + "/info_" + station_id.replace(COUNTRY_ABBREVIATION, "") + ".txt"
    with open(fn, "w") as f:
        print(
            precip.shape[0],
            precip.shape[1],
            precip.shape[2],
            lat_lon_lst[0][0],
            lat_lon_lst[0][1],
            lat_lon_lst[-1][0],
            lat_lon_lst[-1][1],
            file=f,
        )
    print(
        [
            station_id,
            lat_lon_lst[0][0],
            lat_lon_lst[0][1],
            lat_lon_lst[-1][0],
            lat_lon_lst[-1][1],
            precip.shape,
        ]
    )


def check(ERA5_static_data_file_name, station_id):
    df_static_data = pd.read_csv(ERA5_static_data_file_name)
    df_static_data["gauge_id"] = df_static_data["gauge_id"].apply(
        lambda s: s.replace(f"{COUNTRY_ABBREVIATION}_", "")
    )
    basin_static_data = df_static_data[df_static_data["gauge_id"] == str(station_id)]
    basin_static_data = basin_static_data.append(
        [basin_static_data] * 5, ignore_index=True
    )
    print(basin_static_data)


def run_processing_for_single_basin(station_id, basins_data, CAMELS_precip_data_folder_name, output_folder_name):
    print(f"working on station with id: {station_id}")
    station_id = str(station_id).zfill(8)
    basin_data = basins_data[basins_data["hru_id"] == int(station_id)]
    try:
        parse_single_basin_precipitation(
            station_id,
            basin_data,
            CAMELS_precip_data_folder_name,
            output_folder_name)
    except Exception as e:
        print(f"parsing precipitation of basin with id: {station_id} failed with exception: {e}")


def main(use_multiprocessing=True):
    radar_precip_data_folder = "/sci/labs/efratmorin/ranga/FloodMLRan/data/stage4_nc_files/"
    boundaries_file_name = (
            PATH_ROOT + f"/CAMELS_US/HCDN_nhru_final_671.shp")
    output_folder_name = PATH_ROOT + "/CAMELS_US/radar_all_data/"
    basins_data = gpd.read_file(boundaries_file_name)
    station_ids_list = basins_data["hru_id"].tolist()
    if use_multiprocessing:
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_station_id = {
                executor.submit(run_processing_for_single_basin, station_id, basins_data,
                                radar_precip_data_folder, output_folder_name): station_id for
                station_id in station_ids_list}
            for future in concurrent.futures.as_completed(future_to_station_id):
                station_id = future_to_station_id[future]
                print(f"finished with station id: {station_id}")
    else:
        for station_id in station_ids_list:
            run_processing_for_single_basin(station_id, basins_data, radar_precip_data_folder, output_folder_name)


if __name__ == "__main__":
    main(False)
