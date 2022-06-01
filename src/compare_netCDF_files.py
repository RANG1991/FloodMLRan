import netCDF4
import numpy as np
import netCDF4 as nc
from glob import glob
import geopandas as gpd

# def main():
# root_folder = ""
# boundaries_file_name = root_folder + "/HCDN_nhru_final_671.shp"
# ERA5_precip_data_folder_name = root_folder + "/Precipitation/"
# # read the basins' boundaries file using gpd.read_file()
# basins_data = gpd.read_file(boundaries_file_name)
# all_precip_files_ERA5 = glob(ERA5_precip_data_folder_name + "/*.nc")
# for station_id in basins_data["hru_id"]:
#     basin_data = basins_data[basins_data['hru_id'] == int(station_id)]
#     basin_boundaries = basin_data.bounds
#     # get the minimum and maximum longitude and latitude (square boundaries)
#     min_lon = np.squeeze(np.floor((basin_boundaries['minx'].values * 10) / 10))
#     min_lat = np.squeeze(np.floor((basin_boundaries['miny'].values * 10) / 10))
#     max_lon = np.squeeze(np.ceil((basin_boundaries['maxx'].values * 10) / 10))
#     max_lat = np.squeeze(np.ceil((basin_boundaries['maxy'].values * 10) / 10))
#     for precip_file_ERA5:
#
#     station_id = str(station_id).zfill(8)
#     print("working on basin with id: {}".format(station_id))
#     dataset = nc.Dataset(fn)
#     ti = dataset['time'][:]
#     tp = np.asarray(dataset['tp'][:, :,:], dtype=np.float64)
