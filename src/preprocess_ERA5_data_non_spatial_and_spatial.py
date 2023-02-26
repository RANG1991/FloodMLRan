import netCDF4 as nc
import geopandas as gpd
from climata.usgs import InstantValueIO
import pandas as pd
import numpy as np
import datetime
from shapely.geometry import Point
from pathlib import Path
import xarray as xr
import concurrent.futures
import pytz

PATH_ROOT = "../../FloodMLRan/data/ERA5"
COUNTRY_ABBREVIATIONS = ["au", "br", "ca", "cl", "gb", "lamah"]
TEST_PERIOD = ("1989-10-01", "1999-09-30")
TRAINING_PERIOD = ("1999-10-01", "2008-09-30")

GRID_DELTA = 0.25


def check_if_discharge_file_exists(station_id, ERA5_discharge_data_folder_name):
    timezone_file = ERA5_discharge_data_folder_name + "/timezone_" + station_id + ".txt"
    discharge_file = ERA5_discharge_data_folder_name + "/dis_" + station_id + ".csv"
    list_files = [timezone_file, discharge_file]
    all_files_exist = all(Path(file_name).exists() for file_name in list_files)
    return all_files_exist


def check_if_all_percip_files_exist(station_id, output_folder_name, country_abbreviation):
    shape_file = output_folder_name + "/shape_" + station_id.replace(country_abbreviation, "") + ".csv"
    percip24_file = output_folder_name + "/precip24_" + station_id.replace(country_abbreviation, "") + ".csv"
    dis24_file = output_folder_name + "/dis24_" + station_id.replace(country_abbreviation, "") + ".csv"
    data24 = output_folder_name + "/data24_" + station_id.replace(country_abbreviation, "") + ".csv"
    info_file = output_folder_name + "/info_" + station_id.replace(country_abbreviation, "") + ".txt"
    latlon_file = output_folder_name + "/latlon_" + station_id.replace(country_abbreviation, "") + ".csv"
    list_files = [shape_file, percip24_file, dis24_file, data24, info_file, latlon_file]
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


def create_and_write_precipitation_spatial(
        datetimes, ls_spatial, ERA5_static_data_file_name, station_id, output_folder_name, lat_grid, lon_grid,
        lat_lon_lst, country_abbreviation):
    h = len(lat_grid)
    w = len(lon_grid)
    idx_mat = np.zeros((h, w), dtype=bool)
    for lat_i, lon_i in lat_lon_lst:
        i, j = get_index_by_lat_lon(lat_i, lon_i, lat_grid, lon_grid)
        idx_mat[i[0], j[0]] = True
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
    ds["precipitation"] = ds["precipitation"] * idx_mat
    df_static_data = pd.read_csv(ERA5_static_data_file_name)
    df_static_data["gauge_id"] = df_static_data["gauge_id"].apply(
        lambda s: s.replace(f"{country_abbreviation}_", "")
    )
    basin_static_data = df_static_data[df_static_data["gauge_id"] == str(station_id)]
    ds = ds.resample(datetime="1D").sum()
    ds.to_netcdf(path=output_folder_name + "/precip24_spatial_" + station_id.replace(country_abbreviation, "") + ".nc")


def create_and_write_precipitation_mean(datetimes, ls,
                                        ERA5_static_data_file_name, station_id,
                                        output_folder_name, country_abbreviation):
    # convert the precipitation times from UTC (Grinch) to current timezone
    # create a dataframe from the precipitation data and the timedates
    df_percip = pd.DataFrame(data=ls, index=datetimes, columns=["precip"])
    # down sample the precipitation data into 1D (1 day) bins and sum the values falling into the same bin
    df_percip_one_day = df_percip.resample("1D").sum()
    df_percip_one_day = df_percip_one_day.reset_index()
    df_percip_one_day = df_percip_one_day.rename(columns={"index": "date"})
    df_percip_one_day = df_percip_one_day[["date", "precip"]]
    print(df_percip_one_day)
    df_static_data = pd.read_csv(ERA5_static_data_file_name)
    df_static_data["gauge_id"] = df_static_data["gauge_id"].apply(
        lambda s: s.replace(f"{country_abbreviation}_", "")
    )
    basin_static_data = df_static_data[df_static_data["gauge_id"] == str(station_id)]
    df_percip_one_day.to_csv(
        output_folder_name + "/precip24_" + station_id.replace(country_abbreviation, "") + ".csv", float_format="%6.1f"
    )
    return df_percip_one_day


def parse_single_basin_precipitation(
        station_id,
        basin_data,
        discharge_file_name,
        ERA5_data_folder_name,
        discharge_folder_name,
        ERA5_static_data_file_name,
        output_folder_name,
        country_abbreviation
):
    # all_files_exist = check_if_all_percip_files_exist(station_id, output_folder_name)
    # if all_files_exist:
    #     print("all precipitation file of the basin: {} exists".format(station_id))
    #     return
    bounds = basin_data.bounds
    # get the minimum and maximum longitude and latitude (square boundaries)
    min_lon = np.squeeze(np.floor(bounds["minx"].values * 10) / 10)
    min_lat = np.squeeze(np.floor(bounds["miny"].values * 10) / 10)
    max_lon = np.squeeze(np.ceil(bounds["maxx"].values * 10) / 10)
    max_lat = np.squeeze(np.ceil(bounds["maxy"].values * 10) / 10)

    # fn = discharge_folder_name + "/timezone_" + station_id + ".txt"
    # with open(fn, "r") as f:
    #     lines = f.readlines()
    #     utc_offset = int(float(lines[0].strip()))
    #
    # # read the discharge of the required station
    # df_dis = pd.read_csv(discharge_file_name)
    # # convert the columns of year, month, day, hour, minute to datetime and put it as the dataframe index
    # df_dis.index = [
    #     datetime.datetime(
    #         df_dis["year"][i],
    #         df_dis["month"][i],
    #         df_dis["day"][i],
    #         df_dis["hour"][i],
    #         df_dis["minute"][i],
    #     )
    #     for i in range(0, len(df_dis))
    # ]
    # # read the precipitation of the required station
    # # ERA5
    # year_start = df_dis["year"].min() - 1
    # year_end = df_dis["year"].max()
    # print(year_start, year_end)

    offset = get_utc_offset((min_lat + max_lat) / 2)
    list_of_dates_all_years = []
    list_of_total_precipitations_all_years = []
    started_reading_data = False
    for year in range(1988, 2009):
        for month in ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]:
            fn = f"{ERA5_data_folder_name}/tp_{country_abbreviation.upper()}_{year}_{month}.nc"
            try:
                dataset = nc.Dataset(fn)
            except Exception as e:
                print(e)
                continue
            # ti is an array containing the dates as the number of hours since 1900-01-01 00:00
            # e.g. - [780168, 780169, 780170, ...]
            ti = dataset["time"][:]
            if not started_reading_data:
                lon = dataset["longitude"][:]
                lat = dataset["latitude"][:]
                min_lon_array = min_lon - lon
                ind_lon_min = np.where(min_lon_array > 0, min_lon_array, np.inf).argmin()
                max_lon_array = lon - max_lon
                ind_lon_max = np.where(max_lon_array > 0, max_lon_array, np.inf).argmin()
                min_lat_array = lat - min_lat
                ind_lat_min = np.where(min_lat_array > 0, min_lat_array, np.inf).argmin()
                max_lat_array = max_lat - lat
                ind_lat_max = np.where(max_lat_array > 0, max_lat_array, np.inf).argmin()
                started_reading_data = True
            tp = np.asarray(
                dataset["tp"][
                :, ind_lat_max: ind_lat_min + 1, ind_lon_min: ind_lon_max + 1
                ]
            )
            # multiply the precipitation by 1000 to get millimeter instead of meter
            tp[:, :, :] = tp[:, :, :] * 1000
            # zero out any precipitation that is less than 0
            tp[tp < 0] = 0
            # ti is an array containing the dates as the number of
            # hours since 1900-01-01 00:00 so this is the starting date
            starting_date = datetime.datetime.strptime("1900-01-01 00:00", "%Y-%m-%d %H:%M")
            # convert the time to datetime format
            datetimes = [
                (starting_date + datetime.timedelta(hours=int(ti[i])))
                for i in range(0, len(ti))
            ]
            # append the datetimes from specific year to the list of datetimes of all years
            list_of_dates_all_years.append(datetimes)
            # append the precipitations from specific year to the list of precipitations of all years
            list_of_total_precipitations_all_years.append(tp)

    # concatenate the datetimes from all the years
    datetimes = np.concatenate(list_of_dates_all_years, axis=0)
    # concatenate the precipitation data from all the years
    precip = np.concatenate(list_of_total_precipitations_all_years, axis=0)
    # take the mean of the precipitation data spatially (along the latitude and longitude)
    precip_mean_lat_lon = np.mean(precip, axis=(1, 2))
    # create a dataframe from the datetimes
    df_precip_times = pd.DataFrame(data=datetimes, index=datetimes)
    datetimes = df_precip_times.index.to_pydatetime()
    # convert the precipitation data to the correct format by subtracting each hour from its previous hour
    # starting from 1 - this is because the precipitation data is cumulative
    precip_mean_lat_lon_new = []
    precip_new = []
    for i in range(precip_mean_lat_lon.shape[0]):
        if datetimes[i].hour != 1 and i > 0:
            precip_mean_lat_lon_new.append(
                precip_mean_lat_lon[i] - precip_mean_lat_lon[i - 1]
            )
            precip_new.append(precip[i, :, :] - precip[i - 1, :, :])
        else:
            precip_mean_lat_lon_new.append(precip_mean_lat_lon[i])
            precip_new.append(precip[i, :, :])
    ls = [[precip_mean_lat_lon_new[i]] for i in range(0, len(datetimes))]
    ls_precip_new = [precip_new[i] for i in range(0, len(datetimes))]
    datetimes = [time + datetime.timedelta(hours=offset) for time in datetimes]
    df_precip_one_day_non_spatial = create_and_write_precipitation_mean(
        datetimes,
        ls,
        ERA5_static_data_file_name,
        station_id,
        output_folder_name,
        country_abbreviation
    )

    lonb = lon[ind_lon_min:ind_lon_max + 1]
    latb = lat[ind_lat_max:ind_lat_min + 1]
    lslon = [lonb[i] for i in range(0, len(lonb)) for j in range(0, len(latb))]
    lslat = [latb[j] for i in range(0, len(lonb)) for j in range(0, len(latb))]
    lat_lon_lst = []
    for i in range(0, len(lslon)):
        if np.squeeze(basin_data["geometry"].contains(Point(lslon[i], lslat[i]))):
            lat_lon_lst.append([lslat[i], lslon[i]])
    fn = output_folder_name + "/latlon_" + station_id.replace(country_abbreviation, "") + ".csv"
    pd.DataFrame(data=lat_lon_lst, columns=["lat", "lon"]).to_csv(
        fn, index=False, float_format="%6.1f"
    )

    create_and_write_precipitation_spatial(
        datetimes,
        ls_precip_new,
        ERA5_static_data_file_name,
        station_id,
        output_folder_name,
        latb,
        lonb,
        lat_lon_lst,
        country_abbreviation
    )
    # down sample the discharge data into 1D (1 day) bins and take the mean of the values falling into the same bin
    # df_dis_one_day = df_dis.resample("1D").mean()
    # df_dis_one_day = df_dis_one_day.reset_index()
    # df_dis_one_day = df_dis_one_day.rename(columns={"index": "date"})
    # df_dis_one_day = df_dis_one_day[["date", "flow"]]
    # print(df_dis_one_day)
    # # join the two dataframes (precipitation and discharge) by date to get the final dataframe
    # df_joined_non_spatial = df_precip_one_day_non_spatial.merge(
    #     df_dis_one_day, on="date"
    # )
    # df_dis_one_day.to_csv(
    #     output_folder_name + "/dis24_" + station_id + ".csv", float_format="%6.1f"
    # )
    # df_joined_non_spatial.to_csv(
    #     output_folder_name + "/data24_" + station_id + ".csv", float_format="%6.1f"
    # )

    fn = output_folder_name + "/shape_" + station_id.replace(country_abbreviation, "") + ".csv"
    pd.DataFrame(
        data=np.array([(len(datetimes),), (precip.shape[1],), (precip.shape[2],)]).T,
        columns=["time", "lat", "lon"],
        index=[1],
    ).to_csv(fn)
    fn = output_folder_name + "/info_" + station_id.replace(country_abbreviation, "") + ".txt"
    with open(fn, "w") as f:
        print(
            precip.shape[0],
            precip.shape[1],
            precip.shape[2],
            lat[ind_lat_min],
            lat[ind_lat_max],
            lon[ind_lon_min],
            lon[ind_lon_max],
            file=f,
        )
    print(
        [
            station_id,
            lat[ind_lat_min],
            lat[ind_lat_max],
            lon[ind_lon_min],
            lon[ind_lon_max],
            precip.shape,
        ]
    )

    fn = output_folder_name + "/info_" + station_id.replace(country_abbreviation, "") + ".txt"
    with open(fn, "w") as f:
        print(
            precip.shape[0],
            precip.shape[1],
            precip.shape[2],
            lat[ind_lat_min],
            lat[ind_lat_max],
            lon[ind_lon_min],
            lon[ind_lon_max],
            file=f,
        )


def check(ERA5_static_data_file_name, station_id, country_abbreviation):
    df_static_data = pd.read_csv(ERA5_static_data_file_name)
    df_static_data["gauge_id"] = df_static_data["gauge_id"].apply(
        lambda s: s.replace(f"{country_abbreviation}_", "")
    )
    basin_static_data = df_static_data[df_static_data["gauge_id"] == str(station_id)]
    basin_static_data = basin_static_data.append(
        [basin_static_data] * 5, ignore_index=True
    )
    print(basin_static_data)


def run_processing_for_single_basin(station_id, basins_data, ERA5_discharge_data_folder_name,
                                    ERA5_percip_data_folder_name, ERA5_static_data_file_name,
                                    output_folder_name, country_abbreviation):
    print(f"working on station with id: {station_id}")
    station_id = str(station_id).zfill(8)
    basin_data = basins_data[basins_data["gauge_id"] == station_id]
    # try:
    #     parse_single_basin_discharge(station_id, basin_data, ERA5_discharge_data_folder_name)
    # except Exception as e:
    #     print(f"parsing discharge of basin with id: {station_id} failed with exception: {e}")
    #     return
    discharge_file_name = (ERA5_discharge_data_folder_name + "/dis_" + str(station_id) + ".csv")
    try:
        parse_single_basin_precipitation(
            station_id,
            basin_data,
            discharge_file_name,
            ERA5_percip_data_folder_name,
            ERA5_discharge_data_folder_name,
            ERA5_static_data_file_name,
            output_folder_name,
            country_abbreviation
        )
    except Exception as e:
        print(f"parsing precipitation of basin with id: {station_id} failed with exception: {e}")


def main(use_multiprocessing=True):
    for country_abbreviation in COUNTRY_ABBREVIATIONS:
        boundaries_file_name = (PATH_ROOT +
                                f"/Caravan/shapefiles/{country_abbreviation}/{country_abbreviation}_basin_shapes.shp")
        ERA5_static_data_file_name = (
                PATH_ROOT + f"/Caravan/attributes/attributes_hydroatlas_{country_abbreviation}.csv")
        ERA5_percip_data_folder_name = PATH_ROOT + "/Precipitation/"
        ERA5_discharge_data_folder_name = PATH_ROOT + "/Discharge/"
        output_folder_name = PATH_ROOT + "/ERA_5_all_data/"
        # check(ERA5_static_data_file_name, "01031500")
        # return
        # read the basins' boundaries file using gpd.read_file()
        basins_data = gpd.read_file(boundaries_file_name)
        station_ids_list = basins_data["gauge_id"].tolist()
        if use_multiprocessing:
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_to_station_id = {
                    executor.submit(run_processing_for_single_basin, station_id, basins_data,
                                    ERA5_discharge_data_folder_name,
                                    ERA5_percip_data_folder_name,
                                    ERA5_static_data_file_name, output_folder_name): station_id for station_id in
                    station_ids_list}
                for future in concurrent.futures.as_completed(future_to_station_id):
                    station_id = future_to_station_id[future]
                    print(f"finished with station id: {station_id}")
        else:
            for station_id in station_ids_list:
                run_processing_for_single_basin(station_id, basins_data, ERA5_discharge_data_folder_name,
                                                ERA5_percip_data_folder_name, ERA5_static_data_file_name,
                                                output_folder_name, country_abbreviation)


if __name__ == "__main__":
    main(False)
