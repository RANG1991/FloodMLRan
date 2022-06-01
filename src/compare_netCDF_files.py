import numpy as np
import netCDF4 as nc
from glob import glob
import geopandas as gpd
import datetime
import pandas as pd


def main():
    root_folder = ""
    boundaries_file_name = root_folder + "/HCDN_nhru_final_671.shp"
    ERA5_precip_data_folder_name = root_folder + "/Precipitation/"
    Caravan_precip_data_folder_name = root_folder + "/C"
    basins_data = gpd.read_file(boundaries_file_name)
    all_precip_files_ERA5 = glob(ERA5_precip_data_folder_name + "/*.nc")
    for station_id in basins_data["hru_id"]:
        print("working on basin with id: {}".format(station_id))
        basin_data = basins_data[basins_data['hru_id'] == int(station_id)]
        basin_boundaries = basin_data.bounds
        # get the minimum and maximum longitude and latitude (square boundaries)
        min_lon = np.squeeze(np.floor((basin_boundaries['minx'].values * 10) / 10))
        min_lat = np.squeeze(np.floor((basin_boundaries['miny'].values * 10) / 10))
        max_lon = np.squeeze(np.ceil((basin_boundaries['maxx'].values * 10) / 10))
        max_lat = np.squeeze(np.ceil((basin_boundaries['maxy'].values * 10) / 10))
        for filename in all_precip_files_ERA5:
            try:
                dataset = nc.Dataset(filename)
            except Exception as e:
                print(e)
                continue
            # ti is an array containing the dates as the number of hours since 1900-01-01 00:00
            # e.g. - [780168, 780169, 780170, ...]
            ti = dataset['time'][:]
            lon = dataset['longitude'][:]
            lat = dataset['latitude'][:]
            max_lon_array = lon.max()
            min_lon_array = lon.min()
            max_lat_array = lat.max()
            min_lat_array = lat.min()
            ind_lon_min = np.squeeze(np.argwhere(lon == max(min_lon, min_lon_array)))
            ind_lon_max = np.squeeze(np.argwhere(lon == min(max_lon, max_lon_array)))
            ind_lat_min = np.squeeze(np.argwhere(lat == max(min_lat, min_lat_array)))
            ind_lat_max = np.squeeze(np.argwhere(lat == min(max_lat, max_lat_array)))
            tp = np.asarray(dataset['tp'][:, ind_lat_max:ind_lat_min + 1, ind_lon_min:ind_lon_max + 1],
                            dtype=np.float64)
            # convert the time to datetime format and append it to the times array
            times = [
                datetime.datetime.strptime("1900-01-01 00:00", "%Y-%m-%d %H:%M") + datetime.timedelta(hours=int(ti[i]))
                for i in range(0, len(ti))]
            times = np.asarray(times)
            df_percip_times = pd.DataFrame(data=times, index=times)
            datetimes = df_percip_times.index.to_pydatetime()
            ls = [[tp[i]] for i in range(0, len(datetimes))]
            df_percip = pd.DataFrame(data=ls, index=datetimes, columns=['precip'])
