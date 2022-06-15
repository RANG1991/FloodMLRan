import glob
import pathlib
import random
import math
import pandas as pd


def main():
    ERA5_file_dir = "../data/ERA5/ERA_5_all_data/"
    train_basins_ids_file = ERA5_file_dir + "/train_basins_ERA5.txt"
    validation_basins_ids_file = ERA5_file_dir + "/validation_basins_ERA5.txt"
    test_basins_ids_file = ERA5_file_dir + "/test_basins_ERA5.txt"
    files_paths = list(glob.glob(f'{ERA5_file_dir}/data24_*.csv'))
    df_caravan_static_attr = pd.read_csv(r"C:\Users\galun\Desktop\Caravan\attributes\attributes_caravan_us.csv")
    basins_ids_caravan = df_caravan_static_attr["gauge_id"]\
        .apply(lambda x: x.replace("us_", "")).values.tolist()
    basins_ids_list = []
    for file_path in files_paths:
        file_name = pathlib.Path(file_path).name
        basin_id = file_name.replace("data24_", "").replace(".csv", "")
        if basin_id in basins_ids_caravan:
            basins_ids_list.append(basin_id)
    random.shuffle(basins_ids_list)
    with open(train_basins_ids_file, "w") as f:
        basins_ids_list_train = basins_ids_list[:math.floor(len(basins_ids_list) * 0.7)]
        f.write("\n".join(basins_ids_list_train))
    with open(validation_basins_ids_file, "w") as f:
        basins_ids_list_validation = basins_ids_list[math.floor(len(basins_ids_list) * 0.7):
                                                     math.floor(len(basins_ids_list) * 0.85)]
        f.write("\n".join(basins_ids_list_validation))
    with open(test_basins_ids_file, "w") as f:
        basins_ids_list_test = basins_ids_list[:math.floor(len(basins_ids_list) * 0.85):
                                               len(basins_ids_list)]
        f.write("\n".join(basins_ids_list_test))


if __name__ == '__main__':
    main()
