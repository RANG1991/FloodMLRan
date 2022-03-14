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


def main():
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
    create_basins_ids_file("./train_basins.txt", list_train)
    create_basins_ids_file("./val_basins.txt", list_test)
    create_basins_ids_file("./test_basins.txt", list_val)


if __name__ == "__main__":
    main()
