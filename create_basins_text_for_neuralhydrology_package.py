import math
from os import path
import re
import glob
import random

ROOT_DIR_PATH = r"C:\Users\Admin\PycharmProjects\FloodMLRan\Data\CAMELS\usgs_streamflow"


def create_basins_ids_file(file_name, list_ids):
    with open(file_name, "w") as f:
        for basin_id in list_ids:
            f.write(basin_id + "\n")


def create_train_test_val_from_all_basins():
    files_in_dir = glob.glob(ROOT_DIR_PATH + '/**/*.txt', recursive=True)
    basins_ids = []
    for f in files_in_dir:
        f_path = path.basename(f)
        m = re.match(r"(\d+)_streamflow_(.*?)\.txt", f_path)
        if m is not None:
            basins_ids.append(m.group(1))
    list_basins_ids_unique = list(set(basins_ids))
    random.shuffle(list_basins_ids_unique)
    len_list_basins = len(list_basins_ids_unique)
    list_train = list_basins_ids_unique[:math.floor(len_list_basins * 0.7)]
    list_val = list_basins_ids_unique[math.floor(len_list_basins * 0.7):math.floor(len_list_basins * 0.85)]
    list_test = list_basins_ids_unique[math.floor(len_list_basins * 0.85):]
    list_train = sorted(list_train)
    list_test = sorted(list_test)
    list_val = sorted(list_val)
    create_basins_ids_file("data/CAMELS/train_basins_local_computer_Ran.txt", list_train)
    create_basins_ids_file("data/CAMELS/val_basins_local_computer_Ran.txt", list_test)
    create_basins_ids_file("data/CAMELS/test_basins_local_comp_Ran.txt", list_val)


def create_train_test_val_from_531_basins_file():
    with open(
            r"C:\Users\Admin\PycharmProjects\FloodMLRan\neuralhydrology\examples\06-Finetuning\531_basin_list.txt") as f:
        basins_ids = []
        for row in f:
            basins_ids.append(row.strip())
            list_basins_ids_unique = list(set(basins_ids))
            random.shuffle(list_basins_ids_unique)
            len_list_basins = len(list_basins_ids_unique)
            list_train = list_basins_ids_unique[:math.floor(len_list_basins * 0.7)]
            list_val = list_basins_ids_unique[math.floor(len_list_basins * 0.7):math.floor(len_list_basins * 0.85)]
            list_test = list_basins_ids_unique[math.floor(len_list_basins * 0.85):]
            list_train = sorted(list_train)
            list_test = sorted(list_test)
            list_val = sorted(list_val)
            create_basins_ids_file("data/CAMELS/train_basins_local_computer_Ran.txt", list_train)
            create_basins_ids_file("data/CAMELS/val_basins_local_computer_Ran.txt", list_test)
            create_basins_ids_file("data/CAMELS/test_basins_local_comp_Ran.txt", list_val)


def main():
    create_train_test_val_from_531_basins_file()


if __name__ == "__main__":
    main()
