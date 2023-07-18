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

gpd.options.use_pygeos = True


def print_locations_on_world_map(df_locations, color, use_map_axis):
    lon_array = df_locations["gauge_lon"]
    lat_array = df_locations["gauge_lat"]
    df_lat_lon_basins = {"Longitude": lon_array,
                         "Latitude": lat_array}
    df_lat_lon_basins = pd.DataFrame.from_dict(df_lat_lon_basins)
    geometry = [Point(xy) for xy in zip(df_lat_lon_basins['Longitude'], df_lat_lon_basins['Latitude'])]
    gdf = GeoDataFrame(df_lat_lon_basins, geometry=geometry)
    gdf.plot(ax=use_map_axis, marker='o', color=color, markersize=20)


def plot_lon_lat_on_world_map(csv_results_file_with_static_attr):
    df_results = pd.read_csv(csv_results_file_with_static_attr)
    df_results = df_results.select_dtypes(include=[np.number]).dropna(how='all')
    df_results = df_results.fillna(df_results.mean())
    print(df_results.corr())
    df_results["label"] = np.where(df_results['NSE_CNN_LSTM_135'] > df_results['NSE_LSTM_135'], 1, 0)
    df_results_label_is_zero = df_results[df_results["label"] == 0]
    df_results_label_is_one = df_results[df_results["label"] == 1]
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    usa = world[world.name == "United States of America"]
    polygon = box(-127, -85, 175, 85)
    usa = gpd.clip(usa, polygon)
    use_map_axis = usa.plot(figsize=(20, 12))
    print_locations_on_world_map(df_results_label_is_zero, "red", use_map_axis)
    print_locations_on_world_map(df_results_label_is_one, "yellow", use_map_axis)
    plt.savefig(f"analysis_images/plot_lat_lon.png")


def create_accumulated_local_effects_and_shap_values(df_results, clf, scale_features=True):
    X_train = df_results.to_numpy()[:, :-1]
    if scale_features:
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
    clf.fit(X_train, df_results["label"])
    score = clf.score(X_train, df_results["label"])
    print(f"the accuracy score of cls: {clf.__class__} is: {score}")
    ale_clf = ALE(clf.predict, feature_names=df_results.columns[:-1], target_names=["label"])
    exp_clf = ale_clf.explain(X_train)
    plot_ale(exp_clf, n_cols=7, fig_kw={'figwidth': 12, 'figheight': 10})
    plt.savefig("analysis_images/ALE.png")
    plt.clf()
    explainer = shap.Explainer(clf.predict, X_train, feature_names=df_results.columns[:-1])
    shap_values = explainer(X_train)
    shap.summary_plot(shap_values, plot_type='violin')
    # shap.plots.beeswarm(shap_values)
    # shap.plots.bar(shap_values)
    plt.savefig("analysis_images/shap.png")


def process_df_results(df_results, with_std=True):
    df_results = df_results.loc[(df_results['NSE_CNN_LSTM_135'] > 0) | (df_results['NSE_LSTM_135'] > 0)]
    df_results["label"] = df_results['NSE_CNN_LSTM_135'] - df_results['NSE_LSTM_135']
    d2 = np.around(df_results["label"], decimals=5)
    res = wilcoxon(d2, alternative='greater')
    print(res)
    # df_results = df_results[abs(df_results['NSE_CNN_LSTM_135'] - df_results['NSE_LSTM_135']) > 0.01]
    df_results = df_results.drop(columns=['NSE_CNN_LSTM_135', 'NSE_LSTM_135'])
    df_results = df_results.set_index("basin_id")
    df_results = df_results.select_dtypes(include=[np.number]).dropna(how='all')
    df_results = df_results.fillna(df_results.mean())
    if with_std:
        df_results = df_results[CAMELS_dataset.STATIC_ATTRIBUTES_NAMES + ["std"] + ["label"]]
    else:
        df_results = df_results[CAMELS_dataset.STATIC_ATTRIBUTES_NAMES + ["label"]]
    return df_results


def analyse_results_by_decision_tree(df_results, scale_features=True):
    clf = DecisionTreeRegressor(random_state=0, max_depth=7)
    X_train = df_results.to_numpy()[:, :-1]
    if scale_features:
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
    clf.fit(X_train, df_results["label"])
    score = clf.score(X_train, df_results["label"])
    print(f"the accuracy score of cls: {clf.__class__} is: {score}")
    plt.figure(figsize=(25, 20))
    tree.plot_tree(clf, feature_names=df_results.columns[:-1], fontsize=12)
    plt.savefig("analysis_images/decision_tree.png")


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


def create_dataframe_of_std_spatial():
    dataset = create_CAMELS_dataset()
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
            basin_id_to_std[basin_id] = x_spatial.nanstd(axis=1).mean().item()
    df_std = pd.DataFrame(basin_id_to_std.items(), columns=["basin_id", "std"])
    df_std["basin_id"] = df_std["basin_id"].astype(int)
    return df_std


def analyse_features(csv_results_file_with_static_attr, clf_name, with_std=True):
    clf, scale_features = get_clf_from_clf_name(clf_name)
    df_results = pd.read_csv(csv_results_file_with_static_attr)
    if with_std:
        df_std = create_dataframe_of_std_spatial()
        df_results = df_results.merge(df_std, how='inner', on="basin_id")
    df_results = process_df_results(df_results, with_std=with_std)
    df_results.to_csv("analysis_images/check.csv")
    analyse_results_by_decision_tree(df_results, scale_features=scale_features)
    print((df_results.corr()["label"]).sort_values(ascending=False))
    create_accumulated_local_effects_and_shap_values(df_results, clf, scale_features=scale_features)
    importance = get_feature_importance_from_trained_clf(clf, clf_name, df_results, scale_features=scale_features)
    plt.figure(figsize=(25, 20))
    plt.xticks(rotation=90)
    plt.bar([x for x in df_results.columns[:-1]], importance)
    plt.savefig(f"analysis_images/feature_importance_{clf_name}.png")


def create_CAMELS_dataset():
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
        model_name="CNN_LSTM",
        sequence_length_spatial=185,
        create_new_files=False,
        all_stations_ids=sorted(open("../data/spatial_basins_list.txt").read().splitlines()),
        sequence_length=180,
        discharge_str=CAMELS_dataset.DISCHARGE_STR,
        use_all_static_attr=False,
        limit_size_above_1000=True,
        num_basins=None,
        use_only_precip_feature=False,
        run_with_radar_data=False
    )
    return camels_dataset


def create_integrated_gradients(checkpoint_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TWO_LSTM_CNN_LSTM(
        input_dim=32,
        image_height=36, image_width=36,
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
    model.train()
    dataset = create_CAMELS_dataset()
    _, _, xs_non_spatial, xs_spatial, _, _ = dataset[0]
    ig = IntegratedGradients(model)
    xs_non_spatial = xs_non_spatial.to(device)
    xs_spatial = xs_spatial.to(device)
    xs_non_spatial.requires_grad_()
    xs_spatial.requires_grad_()
    feature_names = CAMELS_dataset.DYNAMIC_ATTRIBUTES_NAMES + CAMELS_dataset.STATIC_ATTRIBUTES_NAMES + ["spatial_input"]
    attr, delta = ig.attribute((xs_non_spatial.unsqueeze(0), xs_spatial.unsqueeze(0)), n_steps=256,
                               return_convergence_delta=True)
    attr_non_spatial = attr[0].detach().cpu().numpy()
    attr_spatial = attr[1].detach().cpu().numpy()
    sum_days_attr_spatial = attr_spatial.sum(axis=1)
    non_zero_attr_spatial = sum_days_attr_spatial[np.nonzero(sum_days_attr_spatial)]
    importances = np.concatenate([np.mean(attr_non_spatial, axis=(0, 1)),
                                  np.mean(non_zero_attr_spatial).reshape(1, )])
    for i in range(len(feature_names)):
        print(feature_names[i], ": ", '%.5f' % (importances[i]))
    x_pos = (np.arange(len(feature_names)))
    plt.figure(figsize=(25, 20))
    plt.xticks(rotation=90)
    plt.bar(x_pos, importances, align='center')
    plt.xticks(x_pos, feature_names, wrap=True)
    plt.xlabel("Features")
    plt.title("Average Feature Importance")
    plt.savefig("analysis_images/integrated_gradients.png")


def create_class_activation_maps_explainable(checkpoint_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    model.eval()
    cam_extractor = SmoothGradCAMpp(model.cnn_lstm.cnn.cnn_layers[4], input_shape=(16, 32, 32))
    dataset = create_CAMELS_dataset()
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
        image_basin = cv2.imread(f"/sci/labs/efratmorin/ranga/FloodMLRan"
                                 f"/data/basin_check_precip_images/img_{basin_id}_precip.png")
        image_basin = cv2.resize(image_basin, (50, 50), interpolation=cv2.INTER_CUBIC)
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
    cv2.imwrite(f"./heat_maps/heat_map_all_basins.png", np.vstack(list_rows))


def main():
    plt.rc('font', size=12)
    plot_lon_lat_on_world_map("17775252_17782018_17828539_17832148_17837642.csv")
    # create_class_activation_maps_explainable("../checkpoints/TWO_LSTM_CNN_LSTM_epoch_number_30_size_above_1000.pt")
    create_integrated_gradients("../checkpoints/TWO_LSTM_CNN_LSTM_epoch_number_30_size_above_1000.pt")
    analyse_features("17775252_17782018_17828539_17832148.csv", "random_forest", with_std=True)


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    main()
