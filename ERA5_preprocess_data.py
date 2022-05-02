from os import listdir
from os.path import isfile, join
import pandas as pd
import xarray as xr

TEST_PERIOD = ("1989-10-01", "1999-09-30")
TRAINING_PERIOD = ("1999-10-01", "2008-09-30")


def preprocess_train_test_data(data):
    print(data)
    data_df = data.to_pandas()
    if len(data["time"].data) > 0:
        data_df["tp"] = data_df["tp"].apply(lambda x: x * 1000)
        data_df.set_index("time")
        data_df_one_day = data_df.resample('1D').sum()
        return data_df_one_day
    else:
        return data_df


def preprocess_single_year_file(basin_filename):
    year_dataset = xr.load_dataset(basin_filename)
    test_data = year_dataset.sel(time=slice(*TEST_PERIOD))
    test_data = test_data.drop_dims(["latitude", "longitude"])
    train_data = year_dataset.sel(time=slice(*TRAINING_PERIOD))[["time", "tp"]]
    train_data = train_data.drop_dims(["latitude", "longitude"])
    preprocessed_test_data = preprocess_train_test_data(test_data)
    preprocessed_train_data = preprocess_train_test_data(train_data)
    return preprocessed_train_data, preprocessed_test_data


def preprocess_data(basins_files_dir):
    basins_files = [f for f in listdir(basins_files_dir) if isfile(join(basins_files_dir, f))]
    df_train = pd.DataFrame(columns=["time", "tp"], index=["tp"])
    df_test = pd.DataFrame(columns=["time", "tp"], index=["tp"])
    for i in range(2):
        basin_filename = join(basins_files_dir, basins_files[i])
        preprocess_train_data, preprocessed_test_data = preprocess_single_year_file(basin_filename)
        pd.concat([df_train, preprocess_train_data])
        pd.concat([df_test, preprocess_train_data])
        print(df_test.head())
        print(df_test.head())


def main():
    preprocess_data("./data/ERA-5/")


if __name__ == "__main__":
    main()
