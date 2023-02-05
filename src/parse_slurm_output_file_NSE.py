import re
import pandas as pd
from pathlib import Path


def read_output_file(output_file, basins_ids_to_include=set()):
    dict_nse_basins = {}
    with open(output_file, "r") as f:
        for row in f:
            find = re.search("station with id: (.*?) has nse of: (.*?)\n", row)
            if find:
                basin_id = find.group(1)
                if len(basins_ids_to_include) > 0 and basin_id not in basins_ids_to_include:
                    continue
                basin_nse = float(find.group(2))
                dict_nse_basins[basin_id] = basin_nse
    return dict_nse_basins


def generate_csv_from_output_file(output_file, basins_ids_to_include=set()):
    basins_dict = read_output_file(output_file, basins_ids_to_include=basins_ids_to_include)
    output_file_name = Path(output_file).stem
    pd.DataFrame(basins_dict.values(), columns=['NSE'], index=basins_dict.keys()).to_csv(
        f"./{output_file_name}.csv")
    return basins_dict


def main():
    basins_dict_conv_lstm = generate_csv_from_output_file("./output_file_conv_lstm.txt")
    basins_set_ids_conv_lstm = set(basins_dict_conv_lstm.keys())
    basins_dict_lstm = generate_csv_from_output_file("./output_file_lstm.txt",
                                                     basins_ids_to_include=basins_set_ids_conv_lstm)


if __name__ == "__main__":
    main()
