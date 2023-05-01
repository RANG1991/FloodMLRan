import netCDF4
import unlzw3
from pathlib import Path
import xarray
import os
import gzip


def convert_all_grb_files_to_netCDF(grib_main_folder):
    all_grib_files_paths = Path(grib_main_folder).resolve().rglob("**/ST4*.*")
    files_count = sum([len(files) for r, d, files in os.walk(grib_main_folder)])
    assert len(all_grib_files_paths) == files_count, f"the total number of files in the directory: " \
                                                     f"{files_count} is not equal " \
                                                     f"to the total number of files using rglob: {len(all_grib_files_paths)}"
    for grib_file_path in all_grib_files_paths:
        if grib_file_path.suffix != "grb2":
            if grib_file_path.suffix == ".gz":
                with gzip.open(grib_file_path, 'rb') as f_in:
                    uncompressed_data = f_in.read()
            elif grib_file_path.suffix == ".Z":
                uncompressed_data = unlzw3.unlzw(grib_file_path)
            else:
                raise Exception(f"unknown extension: {grib_file_path.suffix}")
            grib_file_path_for_reading_using_xarray = grib_file_path.parent + grib_file_path.stem + ".grb2"
            with open(grib_file_path_for_reading_using_xarray, "wb") as f_out:
                f_out.write(uncompressed_data)
        else:
            grib_file_path_for_reading_using_xarray = grib_file_path
    data = xarray.open_dataset(grib_file_path_for_reading_using_xarray, engine='cfgrib')
    file_path_net_CDF = grib_file_path_for_reading_using_xarray.parent + grib_file_path_for_reading_using_xarray.stem + ".nc"
    data.to_netcdf(file_path_net_CDF)
    nc_file = netCDF4.Dataset(file_path_net_CDF)
    print(nc_file.variables)


def main():
    convert_all_grb_files_to_netCDF("/sci/labs/efratmorin/ranga/FloodMLRan/data/stage4/")


if __name__ == "__main__":
    main()
