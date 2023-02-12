import re
import pandas as pd
from pathlib import Path
import functools
import numpy as np
from ERA5_dataset import Dataset_ERA5


def read_output_file(output_file):
    dicts_models_dict = {}
    model_name = "empty_model_name"
    with open(output_file, "r", encoding="utf-8") as f:
        for row in f:
            match_model_name_string = re.search("running with model: (.*?)\n", row)
            if match_model_name_string:
                new_model_name = match_model_name_string.group(1)
                if new_model_name != model_name:
                    print(f"moving to new model with name: {new_model_name}")
                    dicts_models_dict[new_model_name] = {}
                    model_name = new_model_name
            match_nse_string = re.search("station with id: (.*?) has nse of: (.*?)\n", row)
            if match_nse_string:
                basin_id = match_nse_string.group(1)
                basin_nse = float(match_nse_string.group(2))
                dicts_models_dict[model_name][basin_id] = basin_nse
    return dicts_models_dict


def generate_box_plots(df_res):
    df_res_new_model_nse_is_higher = df_res.loc[(df_res['NSE_CONV_LSTM'] > df_res['NSE_LSTM'])]
    df_res_new_model_nse_is_lower = df_res.loc[(df_res['NSE_CONV_LSTM'] <= df_res['NSE_LSTM'])]
    for column_name in df_res.columns:
        dict_single_column_box_plot = {"1": df_res_new_model_nse_is_higher[column_name],
                                       "0": df_res_new_model_nse_is_lower[column_name]}
        Dataset_ERA5.create_boxplot_on_data(dict_single_column_box_plot,
                                            plot_title=f"box_plot_{column_name}_NSE_comparison")


def generate_csv_from_output_file(output_file, static_attr_file):
    basins_dict = read_output_file(output_file)
    output_file_name = Path(output_file).stem
    basins_ids = functools.reduce(lambda basin_ids_1, basin_ids_2: basin_ids_1.intersection(basin_ids_2),
                                  [set(basins_dict[model_name].keys()) for model_name in basins_dict.keys()])
    basins_dict_for_data_frame = {}
    for basins_id in basins_ids:
        list_nse_different_models = [basins_id]
        for model_name in basins_dict.keys():
            list_nse_different_models.append(basins_dict[model_name][basins_id])
        basins_dict_for_data_frame[basins_id] = list_nse_different_models[:]
    df_nse = pd.DataFrame.from_dict(basins_dict_for_data_frame, orient="index",
                                    columns=["basin_id"] + [f'NSE_{model_name}' for model_name in basins_dict.keys()])
    df_static_attrib = pd.read_csv(static_attr_file)
    df_static_attrib["gauge_id"] = df_static_attrib["gauge_id"].apply(lambda x: x.replace("us_", ""))
    df_res = df_nse.set_index('basin_id').join(df_static_attrib.set_index('gauge_id'))
    df_res.to_csv(output_file_name + ".csv")


def main():
    generate_csv_from_output_file("./slurm-5703010.out", "../data/ERA5/Caravan/attributes/attributes_hydroatlas_us.csv")


if __name__ == "__main__":
    main()
