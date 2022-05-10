from os import listdir
from os.path import isfile, join
import pandas as pd
import xarray as xr

TEST_PERIOD = ("1989-10-01", "1999-09-30")
TRAINING_PERIOD = ("1999-10-01", "2008-09-30")


def preprocess_train_test_data(data):
    data_df = data.to_pandas()
    if len(data["time"].data) > 0:
        data_df["tp"] = data_df["tp"].apply(lambda x: x * 1000)
        data_df = data_df.reset_index()
        data_df["time"] = pd.to_datetime(data_df["time"], infer_datetime_format=True)
        data_df = data_df.resample('1D', on='time').sum()
        data_df = data_df.reset_index()
    return data_df


def preprocess_single_year_file(basin_filename):
    year_dataset = xr.load_dataset(basin_filename)
    year_dataset = year_dataset.sum(["longitude", "latitude"])
    test_data = year_dataset.sel(time=slice(*TEST_PERIOD))
    train_data = year_dataset.sel(time=slice(*TRAINING_PERIOD))
    preprocessed_test_data = preprocess_train_test_data(test_data)
    preprocessed_train_data = preprocess_train_test_data(train_data)
    return preprocessed_train_data, preprocessed_test_data


def preprocess_data(basins_files_dir):
    basins_files = [f for f in listdir(basins_files_dir) if isfile(join(basins_files_dir, f))]
    df_train = pd.DataFrame(columns=["time", "tp"])
    df_test = pd.DataFrame(columns=["time", "tp"])
    for i in range(len(basins_files)):
        print(f'processing file {i + 1} from {len(basins_files)} files')
        basin_filename = join(basins_files_dir, basins_files[i])
        preprocessed_train_data, preprocessed_test_data = preprocess_single_year_file(basin_filename)
        df_train = pd.concat([df_train, preprocessed_train_data])
        df_test = pd.concat([df_test, preprocessed_test_data])
    df_train = df_train.sort_values(by=["time"], ascending=True)
    df_train.to_csv("./df_train.csv")
    df_test = df_test.sort_values(by=["time"], ascending=True)
    df_test.to_csv("./df_test.csv")


def main():
    preprocess_data("./ERA5/")

