import netCDF4
import unlzw3
from pathlib import Path
import xarray
import os
import gzip
import shutil
import concurrent.futures


def convert_all_grb_files_to_netCDF(grib_file_path, grib2_res_folder_path, nc_res_folder_path):
    print(f"parsing file: {grib_file_path}")
    if not (grib2_res_folder_path / grib_file_path.parent.name).exists():
        os.mkdir(grib2_res_folder_path / grib_file_path.parent.name)
    grib_file_path_for_reading_using_xarray = grib2_res_folder_path / grib_file_path.parent.name / (
            grib_file_path.stem + ".grb2")
    file_path_net_CDF = nc_res_folder_path / grib_file_path.parent.name / (grib_file_path.stem + ".nc")
    if file_path_net_CDF.exists():
        print(f"the netCDF file: {file_path_net_CDF} exists")
        return
    if not grib_file_path_for_reading_using_xarray.exists():
        if grib_file_path.suffix != ".grb2":
            if grib_file_path.suffix == ".gz":
                with gzip.open(grib_file_path, 'rb') as f_in:
                    uncompressed_data = f_in.read()
            elif grib_file_path.suffix == ".Z":
                try:
                    uncompressed_data = unlzw3.unlzw(grib_file_path)
                except ValueError as e:
                    print(e)
                    return
            else:
                raise Exception(f"unknown extension: {grib_file_path}")
            with open(grib_file_path_for_reading_using_xarray, "wb") as f_out:
                f_out.write(uncompressed_data)
        else:
            shutil.copy(grib_file_path, grib_file_path_for_reading_using_xarray)
    data = xarray.open_dataset(grib_file_path_for_reading_using_xarray, engine='cfgrib')
    if not (nc_res_folder_path / grib_file_path.parent.name).exists():
        os.mkdir(nc_res_folder_path / grib_file_path.parent.name)
    data.to_netcdf(file_path_net_CDF)
    nc_file = netCDF4.Dataset(file_path_net_CDF)
    print(f"finished with file: {grib_file_path}")


def main(use_multiprocessing=True):
    grib_main_folder = "/sci/labs/efratmorin/ranga/FloodMLRan/data/stage4/"
    all_grib_files_paths = list(Path(grib_main_folder).resolve().rglob("*.*"))
    # files_count = sum([1 for r, d, files in os.walk(grib_main_folder) for f in files])
    # assert len(all_grib_files_paths) == files_count, f"the total number of files in the directory: " \
    #                                                  f"{files_count} is not equal " \
    #                                                  f"to the total number of files using rglob: " \
    #                                                  f"{len(all_grib_files_paths)}"
    grib2_res_folder_path = Path(grib_main_folder).resolve().parent / "stage4_grib_files"
    nc_res_folder_path = Path(grib_main_folder).resolve().parent / "stage4_nc_files"
    # if not grib2_res_folder_path.exists():
    #     os.mkdir(grib2_res_folder_path)
    # else:
    #     files = grib2_res_folder_path.glob('**/*.*')
    #     for f in files:
    #         os.remove(f)
    # if not nc_res_folder_path.exists():
    #     os.mkdir(nc_res_folder_path)
    # else:
    #     files = nc_res_folder_path.glob('**/*.*')
    #     for f in files:
    #         os.remove(f)
    if use_multiprocessing:
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_grib_file_path = {
                executor.submit(convert_all_grb_files_to_netCDF, grib_file_path, grib2_res_folder_path,
                                nc_res_folder_path): grib_file_path
                for grib_file_path in all_grib_files_paths}
            for future in concurrent.futures.as_completed(future_to_grib_file_path):
                grib_file_path = future_to_grib_file_path[future]
                print(f"finished with grib file: {grib_file_path}")
    else:
        for grib_file_path in all_grib_files_paths:
            convert_all_grb_files_to_netCDF(grib_file_path, grib2_res_folder_path, nc_res_folder_path)


if __name__ == "__main__":
    main(False)
