import re
import pandas as pd
from pathlib import Path
import functools
from ERA5_dataset import Dataset_ERA5
from create_CDF_NSE_comparison_plot_from_slurm_output import calc_dicts_from_all_runs_and_all_files


def generate_box_plots(df_res):
    df_res_new_model_nse_is_higher = df_res.loc[(df_res['NSE_CONV_LSTM'] > df_res['NSE_LSTM'])]
    df_res_new_model_nse_is_lower = df_res.loc[(df_res['NSE_CONV_LSTM'] <= df_res['NSE_LSTM'])]
    for column_name in ["glc_pc_s06",
                        "glc_pc_s09",
                        "tbi_cl_smj",
                        "crp_pc_sse",
                        "glc_pc_s16",
                        "for_pc_sse",
                        "lit_cl_smj",
                        "pnv_pc_s08",
                        "pnv_pc_s09",
                        "pnv_pc_s10",
                        "pnv_pc_s13",
                        "slt_pc_sav",
                        "glc_cl_smj",
                        "inu_pc_slt",
                        "ero_kh_sav",
                        "tec_cl_smj",
                        "snd_pc_sav"]:
        dict_single_column_box_plot = {"new model NSE is higher": df_res_new_model_nse_is_higher[column_name],
                                       "new model NSE is lower": df_res_new_model_nse_is_lower[column_name]}
        Dataset_ERA5.create_boxplot_on_data(dict_single_column_box_plot,
                                            plot_title=f"box_plot_{column_name}_NSE_comparison")


def generate_csv_from_output_file(slurm_output_files, static_attr_file):
    input_file_paths = [Path(f"../slurm_output_files/slurm_files_ensemble_comparison/{file_name}").resolve() for
                        file_name in slurm_output_files]
    dict_all_runs_from_all_files, dict_avg_runs_from_all_files = calc_dicts_from_all_runs_and_all_files(
        input_file_paths)
    input_files_names_formatted = "_".join(
        [input_file_path.name.replace('slurm-', '').replace('.out', '') for input_file_path in input_file_paths])
    output_file_name = Path(input_files_names_formatted).stem
    all_basins_tuples = set([basin_tuple for _, _, basin_tuple, _ in dict_all_runs_from_all_files.keys()])
    basins_ids = [basin_tuple for basin_tuple in all_basins_tuples if len(basin_tuple) == 135][0]
    # for (model_name, params_tuple) in dict_avg_runs_from_all_files.keys():
    # if basins_ids is None:
    #     basins_ids = basins_tuple
    # elif basins_ids != basins_tuple:
    #     raise Exception("not all basins tuples are the same - the CSV will be incorrect")
    basins_dict_for_data_frame = {"basin_id": basins_ids}
    for (model_name, params_tuple) in dict_avg_runs_from_all_files.keys():
        if len(dict_avg_runs_from_all_files[(model_name, params_tuple)]) != 135:
            continue
        basins_dict_for_data_frame[f'NSE_{model_name}_135'] = dict_avg_runs_from_all_files[(model_name, params_tuple)]
    df_nse = pd.DataFrame(basins_dict_for_data_frame)
    df_static_attrib = pd.read_csv(static_attr_file, dtype={"gauge_id": str})
    df_static_attrib["gauge_id"] = df_static_attrib["gauge_id"].apply(lambda x: x.replace("us_", ""))
    df_res = df_nse.set_index('basin_id').join(df_static_attrib.set_index('gauge_id'))
    df_res.to_csv(output_file_name + ".csv")
    # generate_box_plots(df_res)


def generate_csv_from_CAMELS_static_attr_files(static_data_folder):
    attributes_path = Path(static_data_folder)
    txt_files = attributes_path.glob("camels_*.txt")
    # Read-in attributes into one big dataframe
    dfs = []
    for txt_file in txt_files:
        df_temp = pd.read_csv(txt_file, sep=";", header=0, dtype={"gauge_id": str})
        df_temp = df_temp.set_index("gauge_id")
        dfs.append(df_temp)
    df = pd.concat(dfs, axis=1)
    df.index = df.index.astype(str)
    df.to_csv("../data/CAMELS_US/camels_attributes_v2.0/attributes_combined.csv")


def main():
    generate_csv_from_CAMELS_static_attr_files("../data/CAMELS_US/camels_attributes_v2.0")
    generate_csv_from_output_file(
        ["slurm-17775252.out", "slurm-17782018.out", "slurm-17828539.out", "slurm-17832148.out", "slurm-17837642.out"],
        "../data/CAMELS_US/camels_attributes_v2.0/attributes_combined.csv")


if __name__ == "__main__":
    main()
