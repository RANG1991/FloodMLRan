import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn import tree
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import CAMELS_dataset
from sklearn.neighbors import KNeighborsRegressor
from sklearn.inspection import permutation_importance
from alibi.explainers import ALE, plot_ale
from torchcam.methods import SmoothGradCAMpp
from FloodML_2_LSTM_CNN_LSTM import TWO_LSTM_CNN_LSTM
from FloodML_2_Transformer_CNN_Transformer import TWO_Transformer_CNN_Transformer
import torch
from PIL import Image
import cv2
from shapely.geometry import Point
from geopandas import GeoDataFrame
import os

os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
from shapely.geometry import box
from matplotlib import colormaps as cm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import shap
import pickle
from scipy.stats import wilcoxon
from captum.attr import IntegratedGradients
from scipy.stats import mannwhitneyu
import glob
from functools import reduce
from FloodML_Base_Dataset import FloodML_Base_Dataset

gpd.options.use_pygeos = True

dict_dynamic_attributes = {
    "prcp(mm/day)": "Daily precipitation",
    "srad(w/m2)": "Daily short-wave radiation",
    "tmax(c)": "Daily max air temperature",
    "tmin(c)": "Daily min air temperature",
    "vp(pa)": "Daily vapor pressure"
}

dict_static_attributes = {
    "p_mean": "Mean daily precipitation",
    "pet_mean": "Mean daily potential evapotranspiration",
    "aridity": "Ratio of mean PET to mean precipitation",
    "p_seasonality": "Seasonality and timing of precipitation",
    "frac_snow": "Fraction of snow",
    "high_prec_freq": "Frequency of high-precipitation days",
    "high_prec_dur": "Average duration of high-precipitation events",
    "low_prec_freq": "Frequency of dry days",
    "low_prec_dur": "Average duration of dry periods",
    "elev_mean": "Catchment mean elevation",
    "slope_mean": "Catchment mean slope",
    "area_gages2": "Catchment area",
    "frac_forest": "Forest fraction",
    "lai_max": "Maximum monthly mean of leaf area index",
    "lai_diff": "Difference between the max. and min. mean of the leaf area index",
    "gvf_max": "Maximum monthly mean of green vegetation fraction",
    "gvf_diff": "Difference between the min. and max. "
                "monthly mean of the green vegetation fraction",
    "soil_depth_pelletier": "Depth to bedrock (maximum 50 m)",
    "soil_depth_statsgo": "Soil depth (maximum 1.5 m)",
    "soil_porosity": "Volumetric porosity",
    "soil_conductivity": "Saturated hydraulic conductivity",
    "max_water_content": "Maximum water content of the soil",
    "sand_frac": "Fraction of sand in the soil",
    "silt_frac": "Fraction of silt in the soil",
    "clay_frac": "Fraction of clay in the soil",
    "carbonate_rocks_frac": "Fraction of the Carbonate sedimentary rocks",
    "geol_permeability": "Surface permeability (log10)"
}


def print_locations_on_world_map(df_locations, color, use_map_axis):
    lon_array = df_locations["gauge_lon"]
    lat_array = df_locations["gauge_lat"]
    df_lat_lon_basins = {"Longitude": lon_array,
                         "Latitude": lat_array}
    df_lat_lon_basins = pd.DataFrame.from_dict(df_lat_lon_basins)
    geometry = [Point(xy) for xy in zip(df_lat_lon_basins['Longitude'], df_lat_lon_basins['Latitude'])]
    gdf = GeoDataFrame(df_lat_lon_basins, geometry=geometry)
    gdf.plot(ax=use_map_axis, marker='o', color=color, markersize=60)


def plot_lon_lat_on_world_map(csv_results_file_with_static_attr, model_name_for_comparison):
    df_results = pd.read_csv(csv_results_file_with_static_attr)
    df_results = df_results.select_dtypes(include=[np.number]).dropna(how='all')
    df_results = df_results.fillna(df_results.mean())
    # print(df_results.corr())
    df_results["label"] = np.where(df_results[f'NSE_{model_name_for_comparison}_135'] > df_results['NSE_LSTM_135'], 1,
                                   0)
    df_results_label_is_zero = df_results[df_results["label"] == 0]
    df_results_label_is_one = df_results[df_results["label"] == 1]
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    usa = world[world.name == "United States of America"]
    polygon = box(-127, -85, 175, 85)
    usa = gpd.clip(usa, polygon)
    use_map_axis = usa.plot(figsize=(30, 15))
    use_map_axis.set_axis_off()
    print_locations_on_world_map(df_results_label_is_zero, "red", use_map_axis)
    print_locations_on_world_map(df_results_label_is_one, "yellow", use_map_axis)
    plt.title(f"LSTM vs. {model_name_for_comparison.replace('_', '-')}", fontsize=30)
    plt.savefig(f"analysis_images/plot_lat_lon_{model_name_for_comparison}.png")
    # plt.clf()


def create_accumulated_local_effects_and_shap_values(df_results, clf, model_name_for_comparison, scale_features=True):
    X_train = df_results.to_numpy()[:, :-1]
    if scale_features:
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
    clf.fit(X_train, df_results["label"])
    score = clf.score(X_train, df_results["label"])
    print(f"the accuracy score of cls: {clf.__class__} is: {score}")
    ale_clf = ALE(clf.predict, feature_names=df_results.columns[:-1], target_names=["label"])
    exp_clf = ale_clf.explain(X_train)
    fig = plt.figure()
    axes = fig.gca()
    axes.xaxis.get_label().set_fontsize(20)
    plot_ale(exp_clf, n_cols=7, ax=axes, fig_kw={'figwidth': 17, 'figheight': 20})
    plt.title(f"ALE of {model_name_for_comparison.replace('_', '-')}", fontsize=30)
    plt.savefig(f"analysis_images/ALE_{model_name_for_comparison}.png")
    plt.clf()
    explainer = shap.Explainer(clf.predict, X_train, feature_names=df_results.columns[:-1])
    shap_values = explainer(X_train)
    shap.summary_plot(shap_values, plot_type='violin')
    # shap.plots.beeswarm(shap_values)
    # shap.plots.bar(shap_values)
    plt.title(f"SHAP values of {model_name_for_comparison.replace('_', '-')}", fontsize=30)
    plt.savefig(f"analysis_images/shap_{model_name_for_comparison}.png")


def process_df_results(df_results, model_name_for_comparison, with_std=True):
    df_results = df_results.loc[
        (df_results[f'NSE_{model_name_for_comparison}_135'] > 0) | (df_results['NSE_LSTM_135'] > 0)]
    res_wilcoxon_test = []
    df_cols_NSE_LSTM = df_results.filter(regex=r"NSE_LSTM_\d+_(.*?)_135")
    df_cols_NSE_CNN_LSTM = df_results.filter(regex=rf"NSE_{model_name_for_comparison}_\d+_(.*?)_135")
    for ind in range(len(df_cols_NSE_LSTM)):
        nse_list_single_basin_LSTM = df_cols_NSE_LSTM.iloc[ind, :]
        nse_list_single_basin_CNN_LSTM = df_cols_NSE_CNN_LSTM.iloc[ind, :]
        res_wilcoxon_test.append(mannwhitneyu(nse_list_single_basin_LSTM, nse_list_single_basin_CNN_LSTM)[1])
    df_results["mann_whitney_test_res"] = res_wilcoxon_test
    df_results["label"] = df_results[f'NSE_{model_name_for_comparison}_135'] - df_results['NSE_LSTM_135']
    df_results.to_csv(f"analysis_images/df_results_{model_name_for_comparison}.csv")
    df_results = df_results.loc[df_results["mann_whitney_test_res"] <= 0.05]
    # df_results = df_results[abs(df_results['NSE_CNN_LSTM_135'] - df_results['NSE_LSTM_135']) > 0.01]
    df_results = df_results.drop(
        columns=[f'NSE_{model_name_for_comparison}_135', 'NSE_LSTM_135', "mann_whitney_test_res"])
    df_results = df_results.set_index("basin_id")
    df_results = df_results.select_dtypes(include=[np.number]).dropna(how='all')
    df_results = df_results.fillna(df_results.mean())
    if with_std:
        df_results = df_results[CAMELS_dataset.STATIC_ATTRIBUTES_NAMES + ["std"] + ["label"]]
    else:
        df_results = df_results[CAMELS_dataset.STATIC_ATTRIBUTES_NAMES + ["label"]]
    return df_results


def analyse_results_by_decision_tree(df_results, model_name_for_comparison, scale_features=True):
    clf = DecisionTreeRegressor(random_state=0, max_depth=2)
    X_train = df_results.to_numpy()[:, :-1]
    if scale_features:
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
    clf.fit(X_train, df_results["label"])
    score = clf.score(X_train, df_results["label"])
    print(f"the accuracy score of cls: {clf.__class__} is: {score}")
    # plt.figure(figsize=(25, 20))
    tree.plot_tree(clf, feature_names=df_results.columns[:-1], fontsize=20)
    plt.title(f"Decision tree of {model_name_for_comparison.replace('_', '-')}", fontsize=30)
    plt.savefig(f"analysis_images/decision_tree_{model_name_for_comparison}.png")


def get_clf_from_clf_name(clf_name):
    scale_features = False
    if clf_name == "decision_tree":
        clf = DecisionTreeRegressor(random_state=0, max_depth=7)
    elif clf_name == "random_forest":
        clf = RandomForestRegressor(random_state=0, max_depth=7)
    elif clf_name == "linear_regression":
        clf = LinearRegression()
        scale_features = True
    elif clf_name == "KNN":
        clf = KNeighborsRegressor()
    else:
        raise Exception(f"unknown cls name: {clf_name}")
    return clf, scale_features


def get_feature_importance_from_trained_clf(clf, clf_name, df_results, scale_features=True):
    X_train = df_results.to_numpy()[:, :-1]
    if scale_features:
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
    clf.fit(X_train, df_results["label"])
    if clf_name == "decision_tree":
        importance = clf.feature_importances_
    elif clf_name == "random_forest":
        importance = clf.feature_importances_
    elif clf_name == "linear_regression":
        importance = clf.coef_[0]
    elif clf_name == "KNN_cls":
        results = permutation_importance(clf, df_results.iloc[:, :-1].to_numpy(),
                                         df_results.iloc[:, -1].to_numpy(),
                                         scoring='accuracy')
        importance = results.importances_mean
    else:
        raise Exception(f"unknown cls name: {clf_name}")
    return importance


def create_dataframe_of_std_spatial(model_name):
    dataset = create_CAMELS_dataset(model_name=model_name)
    lookup_table = dataset.lookup_table
    dataset_length = len(dataset)
    curr_basin_id = -1
    basin_id_to_indices = {}
    for ind in range(dataset_length):
        basin_id, _ = lookup_table[ind]
        if basin_id != curr_basin_id:
            basin_id_to_indices[basin_id] = ind
            curr_basin_id = basin_id
    basin_id_to_std = {}
    for basin_id in basin_id_to_indices.keys():
        print(f"in basin: {basin_id}")
        with open(f"{dataset.folder_with_basins_pickles}/{basin_id}_{dataset.stage}{dataset.suffix_pickle_file}.pkl",
                  'rb') as f:
            dict_curr_basin = pickle.load(f)
            x_spatial = dict_curr_basin["x_data_spatial"]
            x_spatial[:, x_spatial.sum(axis=0) <= 0] = np.nan
            basin_id_to_std[basin_id] = np.nanstd(x_spatial, axis=1).mean().item()
    df_std = pd.DataFrame(basin_id_to_std.items(), columns=["basin_id", "std"])
    df_std["basin_id"] = df_std["basin_id"].astype(int)
    return df_std


def analyse_features(csv_results_file_with_static_attr, clf_name, model_name_for_comparison, with_std=True):
    clf, scale_features = get_clf_from_clf_name(clf_name)
    df_results = pd.read_csv(csv_results_file_with_static_attr)
    if with_std:
        df_std = create_dataframe_of_std_spatial(model_name=model_name_for_comparison)
        df_results = df_results.merge(df_std, how='inner', on="basin_id")
    df_results = process_df_results(df_results, model_name_for_comparison, with_std=with_std)
    analyse_results_by_decision_tree(df_results, model_name_for_comparison, scale_features=scale_features)
    corr_df = (df_results.corr(method='pearson')["label"]).sort_values(ascending=False)
    corr_df = corr_df.loc[(np.abs(corr_df) > 0.2)]
    create_accumulated_local_effects_and_shap_values(df_results, clf, model_name_for_comparison,
                                                     scale_features=scale_features)
    importance = get_feature_importance_from_trained_clf(clf, clf_name, df_results, scale_features=scale_features)
    plt.clf()
    # plt.figure(figsize=(25, 20))
    plt.xticks(rotation=90)
    plt.bar([x for x in df_results.columns[:-1]], importance)
    plt.title(f"{model_name_for_comparison.replace('_', '-')}", fontsize=30)
    plt.title(f"Feature importance of {model_name_for_comparison.replace('_', '-')}", fontsize=30)
    plt.savefig(f"analysis_images/feature_importance_{clf_name}_{model_name_for_comparison}.png")
    corr_df = corr_df.rename(index=dict_static_attributes)
    print(corr_df.to_latex(float_format="{:.2f}".format).replace("\\\n", "\\\n\hline\n"))
    return corr_df, df_results


def create_CAMELS_dataset(model_name):
    camels_dataset = CAMELS_dataset.Dataset_CAMELS(
        main_folder=CAMELS_dataset.MAIN_FOLDER,
        dynamic_data_folder=CAMELS_dataset.DYNAMIC_DATA_FOLDER_NON_SPATIAL,
        static_data_folder=CAMELS_dataset.STATIC_DATA_FOLDER,
        dynamic_data_folder_spatial=CAMELS_dataset.DYNAMIC_DATA_FOLDER_SPATIAL_CAMELS,
        discharge_data_folder=CAMELS_dataset.DISCHARGE_DATA_FOLDER,
        dynamic_attributes_names=CAMELS_dataset.DYNAMIC_ATTRIBUTES_NAMES,
        static_attributes_names=CAMELS_dataset.STATIC_ATTRIBUTES_NAMES,
        train_start_date="01/10/1999",
        train_end_date="30/09/2008",
        validation_start_date="01/10/1988",
        validation_end_date="30/09/1993",
        test_start_date="01/10/1993",
        test_end_date="30/09/1999",
        stage="train",
        model_name=model_name,
        sequence_length_spatial=185,
        create_new_files=False,
        all_stations_ids=sorted(open("../data/spatial_basins_list.txt").read().splitlines()),
        sequence_length=180,
        discharge_str=CAMELS_dataset.DISCHARGE_STR,
        use_all_static_attr=False,
        limit_size_above_1000=True,
        num_basins=None,
        use_only_precip_feature=False,
        run_with_radar_data=False,
    )
    return camels_dataset


def create_integrated_gradients(checkpoint_path, model_name_for_comparison, df_results):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_name_for_comparison == "CNN_Transformer":
        model = TWO_Transformer_CNN_Transformer(image_width=36,
                                                image_height=36,
                                                num_static_attributes=27,
                                                num_dynamic_attributes=5,
                                                sequence_length_cnn_transformer=185,
                                                intermediate_dim_transformer=64,
                                                dropout=0.4,
                                                num_heads_transformer=4,
                                                num_layers_transformer=8,
                                                in_cnn_channels=1)
    else:
        model = TWO_LSTM_CNN_LSTM(input_dim=32,
                                  image_height=36,
                                  image_width=36,
                                  hidden_dim=256,
                                  sequence_length_conv_lstm=185,
                                  in_cnn_channels=1,
                                  dropout=0.4,
                                  num_static_attributes=27,
                                  num_dynamic_attributes=5,
                                  use_only_precip_feature=False)
    model = model.to(device=device)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    # model = model.train()
    dataset = create_CAMELS_dataset(model_name=model_name_for_comparison)
    _, _, xs_non_spatial, xs_spatial, _, _ = dataset[0]
    basin_id, _ = dataset.lookup_table[0]
    ig = IntegratedGradients(model)
    xs_non_spatial = xs_non_spatial.to(device)
    xs_spatial = xs_spatial.to(device)
    xs_non_spatial.requires_grad_()
    xs_spatial.requires_grad_()
    feature_names_static = [dict_static_attributes[feature_name_static] for feature_name_static in
                            dataset.list_static_attributes_names]
    feature_names_dynamic = [dict_dynamic_attributes[feature_name_dynamic] for feature_name_dynamic in
                             dataset.list_dynamic_attributes_names]
    feature_names = feature_names_dynamic + feature_names_static + ["spatial input"]
    baseline_values_non_spatial = (
        torch.cat(
            [torch.tensor([0.0]),
             torch.tensor(dataset.x_means[1:len(dataset.list_dynamic_attributes_names)]),
             torch.tensor(df_results.loc[:, dataset.list_static_attributes_names].mean())]
        ).to(device))
    baseline_values_non_spatial = (torch.ones_like(xs_non_spatial) * baseline_values_non_spatial).to(torch.float32)
    baseline_values_spatial = torch.zeros_like(xs_spatial).to(device).to(torch.float32)
    attr, delta = ig.attribute((xs_non_spatial.unsqueeze(0), xs_spatial.unsqueeze(0)),
                               n_steps=100,
                               method='gausslegendre',
                               return_convergence_delta=True,
                               )
    attr_non_spatial = attr[0].detach().cpu().numpy()
    attr_spatial = attr[1].detach().cpu().numpy()
    sum_days_attr_spatial = attr_spatial.sum(axis=1)
    non_zero_attr_spatial = sum_days_attr_spatial[np.nonzero(sum_days_attr_spatial)]
    importances = np.concatenate([np.mean(attr_non_spatial, axis=(0, 1)),
                                  np.mean(non_zero_attr_spatial).reshape(1, )])
    x_pos = (np.arange(len(feature_names)))
    plt.figure(figsize=(12, 19))
    plt.xticks(rotation=90)
    plt.bar(x_pos, importances, align='center')
    plt.xticks(x_pos, feature_names)
    plt.xlabel("Features")
    plt.title("Average Feature Importance")
    plt.savefig(f"./analysis_images/integrated_gradients_{model_name_for_comparison}.png")


def create_class_activation_maps_explainable(checkpoint_path, model_name_for_comparison):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_name_for_comparison == "CNN_Transformer":
        model = TWO_Transformer_CNN_Transformer(image_width=36,
                                                image_height=36,
                                                num_static_attributes=27,
                                                num_dynamic_attributes=5,
                                                sequence_length_cnn_transformer=185,
                                                intermediate_dim_transformer=64,
                                                dropout=0.4,
                                                num_heads_transformer=4,
                                                num_layers_transformer=8,
                                                in_cnn_channels=1)
    else:
        model = TWO_LSTM_CNN_LSTM(
            input_dim=32,
            image_height=36, image_width=36,
            hidden_dim=256,
            sequence_length_conv_lstm=185,
            in_cnn_channels=1,
            dropout=0,
            num_static_attributes=27,
            num_dynamic_attributes=5,
            use_only_precip_feature=False)
    model = model.to(device=device)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.eval()
    if model_name_for_comparison == "CNN_Transformer":
        cam_extractor = SmoothGradCAMpp(model.cnn_transformer.cnn.cnn_layers[4], input_shape=(16, 32, 32))
    else:
        cam_extractor = SmoothGradCAMpp(model.cnn_lstm.cnn.cnn_layers[4], input_shape=(16, 32, 32))
    dataset = create_CAMELS_dataset(model_name=model_name_for_comparison)
    lookup_table = dataset.lookup_table
    dataset_length = len(dataset)
    curr_basin_id = -1
    basin_id_to_first_ind = {}
    for ind in range(dataset_length):
        basin_id, _ = lookup_table[ind]
        if basin_id != curr_basin_id:
            basin_id_to_first_ind[basin_id] = ind
            curr_basin_id = basin_id
    list_all_images = []
    for basin_id in basin_id_to_first_ind.keys():
        print(f"in basin: {basin_id}")
        _, _, xs_non_spatial, xs_spatial, _, _ = dataset[basin_id_to_first_ind[basin_id]]
        out = model(xs_non_spatial.unsqueeze(0).to(device), xs_spatial.unsqueeze(0).to(device))
        activation_map = cam_extractor(0, out.item())
        plt.axis('off')
        plt.tight_layout()
        image_basin = cv2.resize(xs_spatial, (50, 50), interpolation=cv2.INTER_CUBIC)
        cmap_image_precip = plt.get_cmap("binary")
        cmap_image_activation = plt.get_cmap("jet")
        # image_basin = cmap_image_precip(image_basin[:, :, :3])
        image_activation = cv2.resize(activation_map[0].cpu().numpy().mean(axis=0), (50, 50),
                                      interpolation=cv2.INTER_CUBIC)
        # image_activation = ((image_activation - image_activation.min())
        #                     / (image_activation.max() - image_activation.min()))
        image_activation = (255 * (cmap_image_activation(image_activation)[:, :, :3])).astype(np.uint8)
        opacity = 0.7
        overlay = (opacity * image_basin + (1 - opacity) * image_activation)
        image_basin_with_margin = cv2.copyMakeBorder(image_basin, 10, 10, 10, 10, cv2.BORDER_CONSTANT,
                                                     value=(255, 255, 255))
        image_activation_with_margin = cv2.copyMakeBorder(image_activation, 10, 10, 10, 10, cv2.BORDER_CONSTANT,
                                                          value=(255, 255, 255))
        list_all_images.append(np.hstack([image_basin_with_margin, image_activation_with_margin]))
        fig = plt.figure(figsize=(32, 16))
        ax1 = fig.add_subplot(121)
        ax1.axis('off')
        FloodML_Base_Dataset.create_color_bar_for_precip_image(precip_image=image_basin, ax=ax1)
        ax2 = fig.add_subplot(122)
        ax2.axis('off')
        FloodML_Base_Dataset.create_color_bar_for_precip_image(precip_image=image_activation, ax=ax2)
        plt.savefig(f"./heat_maps/heat_map_basin_{basin_id}.png")
    list_images_row = []
    list_rows = []
    num_images_in_row = 10
    for i in range(len(list_all_images)):
        list_images_row.append(list_all_images[i])
        if i % num_images_in_row == (num_images_in_row - 1) or (i == len(list_all_images) - 1):
            list_rows.append(np.hstack(list_images_row))
            list_images_row = []
    white_image = 255 * np.ones((50, 50, 3))
    white_image_with_margin = cv2.copyMakeBorder(white_image, 10, 10, 10, 10, cv2.BORDER_CONSTANT,
                                                 value=(255, 255, 255))
    num_white_images_to_fill = len(list_all_images) % num_images_in_row
    white_images_to_fill = np.hstack([white_image_with_margin for _ in range(num_white_images_to_fill * 2)])
    list_rows[-1] = np.hstack([list_rows[-1], white_images_to_fill])
    cv2.imwrite(f"./analysis_images/heat_map_all_basins_{model_name_for_comparison}.png", np.vstack(list_rows))


def plot_images_side_by_side(models_names):
    all_images_files = [file for file in glob.glob(f"./analysis_images/*.png")]
    all_images_pairs = {}
    first_model_name_for_comparison = models_names[0]
    for image_file_name in all_images_files:
        if first_model_name_for_comparison in image_file_name:
            image_pair = [(first_model_name_for_comparison, image_file_name)]
            for model_name_for_comparison in models_names[1:]:
                image_file_name_with_model_name = image_file_name.replace(first_model_name_for_comparison,
                                                                          model_name_for_comparison)
                image_pair.append((model_name_for_comparison, image_file_name_with_model_name))
            all_images_pairs[image_file_name.replace(f"_{first_model_name_for_comparison}.png", "")] = image_pair
    for key in all_images_pairs.keys():
        curr_image_pair = all_images_pairs[key]
        f, axarr = plt.subplots(len(curr_image_pair), 1, figsize=(24, 38))
        for j in range(len(curr_image_pair)):
            model_name_for_comparison, image_file_name_with_model_name = curr_image_pair[j]
            axarr[j].imshow(plt.imread(image_file_name_with_model_name))
            axarr[j].axis("off")
            # axarr[j].set_title(f"{model_name_for_comparison.replace('_', '-')}", size=30)
        f.tight_layout()
        # f.tight_layout(rect=[0.0, 0.0, 0.1, 0.1])
        f.savefig(f"{key}.png")


def main():
    model_names = [
        "CNN_LSTM",
        # "CNN_Transformer"
    ]
    for model_name_for_comparison in model_names:
        if model_name_for_comparison == "CNN_Transformer":
            checkpoint = "TWO_Transformer_CNN_Transformer_epoch_number_30_size_above_1000"
        else:
            checkpoint = "TWO_LSTM_CNN_LSTM_epoch_number_30_size_above_1000"
        print(f"analysing {model_name_for_comparison}")
        plt.rc('font', size=14)
        plt.rcParams["figure.figsize"] = (12, 19)
        plt.rcParams["figure.autolayout"] = True
        # plot_lon_lat_on_world_map("17775252_17782018_17828539_17832148_17837642_18941386.csv",
        #                           model_name_for_comparison)
        # corr_df, df_results = analyse_features("17775252_17782018_17828539_17832148_17837642_18941386.csv",
        #                                        "random_forest", model_name_for_comparison, with_std=False)
        create_class_activation_maps_explainable(f"../{checkpoint}.pt", model_name_for_comparison)
        # create_integrated_gradients(f"../{checkpoint}.pt", model_name_for_comparison, df_results)

    plot_images_side_by_side(model_names)


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    main()
