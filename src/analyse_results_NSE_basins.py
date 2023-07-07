import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn import tree
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import CAMELS_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
from alibi.explainers import ALE, plot_ale
from torchcam.methods import SmoothGradCAMpp
from FloodML_2_LSTM_CNN_LSTM import TWO_LSTM_CNN_LSTM
import torch
import os
from PIL import Image
from matplotlib import cm
import cv2
from FloodML_Base_Dataset import FloodML_Base_Dataset
from shapely.geometry import Point
from shapely.geometry import Polygon
from geopandas import GeoDataFrame
import geopandas as gpd
from shapely.geometry import box


def print_locations_on_world_map(df_locations, color, use_map_axis):
    lon_array = df_locations["gauge_lon"]
    lat_array = df_locations["gauge_lat"]
    df_lat_lon_basins = {"Longitude": lon_array,
                         "Latitude": lat_array}
    df_lat_lon_basins = pd.DataFrame.from_dict(df_lat_lon_basins)
    geometry = [Point(xy) for xy in zip(df_lat_lon_basins['Longitude'], df_lat_lon_basins['Latitude'])]
    gdf = GeoDataFrame(df_lat_lon_basins, geometry=geometry)
    gdf.plot(ax=use_map_axis, marker='o', color=color, markersize=8)


def plot_lon_lat_on_world_map(csv_results_file_with_static_attr):
    df_results = pd.read_csv(csv_results_file_with_static_attr)
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
    plt.savefig(f"plot_lat_lon.png")


def create_accumulated_local_effects(clf, df_results):
    fun_clf = clf.predict_proba
    ale_clf = ALE(fun_clf, feature_names=CAMELS_dataset.STATIC_ATTRIBUTES_NAMES, target_names=["label"])
    exp_clf = ale_clf.explain(df_results.to_numpy()[:, :-1])
    plot_ale(exp_clf, n_cols=7, fig_kw={'figwidth': 12, 'figheight': 10})
    plt.savefig("ALE.png")


def fit_clf_analysis(csv_results_file_with_static_attr, clf):
    df_results = pd.read_csv(csv_results_file_with_static_attr)
    df_results["label"] = np.where(df_results['NSE_CNN_LSTM_135'] > df_results['NSE_LSTM_135'], 1, 0)
    df_results = df_results.drop(columns=['NSE_CNN_LSTM_135', 'NSE_LSTM_135'])
    df_results = df_results.set_index("basin_id")
    df_results = df_results.select_dtypes(include=[np.number]).dropna()
    df_results = df_results[CAMELS_dataset.STATIC_ATTRIBUTES_NAMES + ["label"]]
    clf.fit(df_results.to_numpy()[:, :-1], df_results["label"])
    return clf, df_results


def analyse_results_by_decision_tree(csv_results_file_with_static_attr):
    clf = DecisionTreeClassifier(random_state=0, max_depth=3)
    clf, df_results = fit_clf_analysis(csv_results_file_with_static_attr, clf)
    plt.figure(figsize=(14, 10))
    tree.plot_tree(clf, feature_names=df_results.columns[:-1],
                   class_names=["0", "1"], fontsize=12)
    plt.savefig("decision_tree.png")


def analyse_results_feat_importance_by_decision_tree(csv_results_file_with_static_attr):
    clf = DecisionTreeClassifier(random_state=0, max_depth=3)
    clf, df_results = fit_clf_analysis(csv_results_file_with_static_attr, clf)
    importance = clf.feature_importances_
    plt.figure(figsize=(25, 20))
    plt.xticks(rotation=90)
    plt.bar([x for x in df_results.columns[:-1]], importance)
    plt.savefig("analysis_decision_tree.png")


def analyse_results_feat_importance_by_logistic_regression(csv_results_file_with_static_attr):
    clf = LogisticRegression()
    clf, df_results = fit_clf_analysis(csv_results_file_with_static_attr, clf)
    create_accumulated_local_effects(clf, df_results)
    importance = clf.coef_[0]
    plt.figure(figsize=(25, 20))
    plt.xticks(rotation=90)
    plt.bar([x for x in df_results.columns[:-1]], importance)
    plt.savefig("analysis_logistic_regression.png")


def analyse_results_feat_importance_by_random_forest(csv_results_file_with_static_attr):
    clf = RandomForestClassifier()
    clf, df_results = fit_clf_analysis(csv_results_file_with_static_attr, clf)
    importance = clf.feature_importances_
    plt.figure(figsize=(25, 20))
    plt.xticks(rotation=90)
    plt.bar([x for x in df_results.columns[:-1]], importance)
    plt.savefig("analysis_random_forest.png")


def analyse_results_feat_importance_by_permutation(csv_results_file_with_static_attr):
    clf = KNeighborsClassifier()
    clf, df_results = fit_clf_analysis(csv_results_file_with_static_attr, clf)
    results = permutation_importance(clf, df_results.iloc[:, :-1].to_numpy(), df_results.iloc[:, -1].to_numpy(),
                                     scoring='accuracy')
    importance = results.importances_mean
    plt.figure(figsize=(25, 20))
    plt.xticks(rotation=90)
    plt.bar([x for x in df_results.columns[:-1]], importance)
    plt.savefig("analysis_permutation.png")


def create_CAMELS_dataset():
    camels_dataset = CAMELS_dataset.Dataset_CAMELS(
        main_folder=CAMELS_dataset.MAIN_FOLDER,
        dynamic_data_folder=CAMELS_dataset.DYNAMIC_DATA_FOLDER_NON_SPATIAL,
        static_data_folder=CAMELS_dataset.STATIC_DATA_FOLDER,
        dynamic_data_folder_spatial=CAMELS_dataset.DYNAMIC_DATA_FOLDER_SPATIAL_CAMELS,
        discharge_data_folder=CAMELS_dataset.DISCHARGE_DATA_FOLDER,
        dynamic_attributes_names=CAMELS_dataset.DYNAMIC_ATTRIBUTES_NAMES,
        static_attributes_names=CAMELS_dataset.STATIC_ATTRIBUTES_NAMES,
        train_start_date="01/10/1997",
        train_end_date="30/09/2002",
        validation_start_date="01/10/1988",
        validation_end_date="30/09/1992",
        test_start_date="01/10/1992",
        test_end_date="30/09/1997",
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


def create_class_activation_maps_explainable(checkpoint_path):
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
    model = model.to(device="cuda")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    cam_extractor = SmoothGradCAMpp(model.cnn_lstm.cnn, input_shape=(1, 36, 36))
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
    for basin_id in basin_id_to_first_ind.keys():
        _, _, xs_non_spatial, xs_spatial, _, _ = dataset[basin_id_to_first_ind[basin_id]]
        out = model(xs_non_spatial.unsqueeze(0).cuda(), xs_spatial.unsqueeze(0).cuda())
        activation_map = cam_extractor(0, out.item())
        plt.axis('off')
        plt.tight_layout()
        cmap_image_precip = cm.get_cmap("binary")
        cmap_image_activation = cm.get_cmap("jet")
        image_precip = (255 * (xs_spatial.cpu().numpy().reshape(xs_spatial.shape[0], 36, 36).mean(axis=0))).astype(
            np.uint8)
        _, _ = cv2.findContours(image_precip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        image_precip = cmap_image_precip(image_precip)[:, :, :3]
        image_activation = (255 * cv2.resize(activation_map[0].cpu().numpy().mean(axis=0), (36, 36),
                                             interpolation=cv2.INTER_CUBIC)).astype(np.uint8)
        image_activation = (cmap_image_activation(
            ((image_activation - image_activation.min()) / (image_activation.max() - image_activation.min())))[:, :,
                            :3])
        opacity = 0.7
        overlay = (opacity * image_precip + (1 - opacity) * image_activation)
        plt.imsave(f"./heat_maps/heat_map_basin_{basin_id}.png", overlay)


def main():
    plot_lon_lat_on_world_map("17775252_17782018_17828539.csv")
    # create_class_activation_maps_explainable(
    #     "/sci/labs/efratmorin/ranga/FloodMLRan/checkpoints/TWO_LSTM_CNN_LSTM_epoch_number_30_size_above_1000.pt")
    # plt.rc('font', size=12)
    # analyse_results_by_decision_tree("17476442_17477923.csv")
    # analyse_results_feat_importance_by_logistic_regression("17476442_17477923.csv")
    # analyse_results_feat_importance_by_decision_tree("17476442_17477923.csv")
    # analyse_results_feat_importance_by_random_forest("17476442_17477923.csv")
    # analyse_results_feat_importance_by_permutation("17476442_17477923.csv")


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
