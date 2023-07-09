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
from PIL import Image
import cv2
from shapely.geometry import Point
from geopandas import GeoDataFrame
import geopandas as gpd
from shapely.geometry import box
from matplotlib import colormaps as cm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import shap

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
    plt.savefig(f"plot_lat_lon.png")


def create_accumulated_local_effects_and_shap_values(df_results, clf):
    clf.fit(df_results.to_numpy()[:, :-1], df_results["label"])
    ale_clf = ALE(clf.predict, feature_names=CAMELS_dataset.STATIC_ATTRIBUTES_NAMES + ["std"], target_names=["label"])
    exp_clf = ale_clf.explain(df_results.to_numpy()[:, :-1])
    plot_ale(exp_clf, n_cols=7, fig_kw={'figwidth': 12, 'figheight': 10})
    plt.savefig("ALE.png")
    plt.clf()
    explainer = shap.Explainer(clf.predict, df_results.to_numpy()[:, :-1],
                               feature_names=CAMELS_dataset.STATIC_ATTRIBUTES_NAMES + ["std"])
    shap_values = explainer(df_results.to_numpy()[:, :-1])
    shap.summary_plot(shap_values, plot_type='violin')
    # shap.plots.bar(shap_values[0])
    plt.savefig("shap.png")


def process_df_results(df_results):
    df_results["label"] = np.where(df_results['NSE_CNN_LSTM_135'] > df_results['NSE_LSTM_135'], 1, 0)
    df_results = df_results.drop(columns=['NSE_CNN_LSTM_135', 'NSE_LSTM_135'])
    df_results = df_results.set_index("basin_id")
    df_results = df_results.select_dtypes(include=[np.number]).dropna(how='all')
    df_results = df_results.fillna(df_results.mean())
    df_results = df_results[CAMELS_dataset.STATIC_ATTRIBUTES_NAMES + ["std"] + ["label"]]
    return df_results


def analyse_results_by_decision_tree(df_results):
    clf = DecisionTreeClassifier(random_state=0, max_depth=1)
    X_train = df_results.to_numpy()[:, :-1]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    clf.fit(X_train, df_results["label"])
    score = accuracy_score(clf.predict(X_train), df_results["label"])
    print(f"the accuracy score of cls: {clf.__class__} is: {score}")
    plt.figure(figsize=(14, 10))
    tree.plot_tree(clf, feature_names=df_results.columns[:-1],
                   class_names=["0", "1"], fontsize=12)
    plt.savefig("decision_tree.png")


def get_clf_from_clf_name(clf_name):
    if clf_name == "decision_tree":
        clf = DecisionTreeClassifier(random_state=0, max_depth=7)
    elif clf_name == "random_forest":
        clf = RandomForestClassifier(random_state=0, max_depth=7)
    elif clf_name == "logistic_regression":
        clf = LogisticRegression(max_iter=10000)
    elif clf_name == "KNN_cls":
        clf = KNeighborsClassifier()
    else:
        raise Exception(f"unknown cls name: {clf_name}")
    return clf


def get_feature_importance_from_trained_clf(clf, clf_name, df_results):
    X_train = df_results.to_numpy()[:, :-1]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    clf.fit(X_train, df_results["label"])
    if clf_name == "decision_tree":
        importance = clf.feature_importances_
    elif clf_name == "random_forest":
        importance = clf.feature_importances_
    elif clf_name == "logistic_regression":
        importance = clf.coef_[0]
    elif clf_name == "KNN_cls":
        results = permutation_importance(clf, df_results.iloc[:, :-1].to_numpy(),
                                         df_results.iloc[:, -1].to_numpy(),
                                         scoring='accuracy')
        importance = results.importances_mean
    else:
        raise Exception(f"unknown cls name: {clf_name}")
    return importance


def analyse_results_feat_importance_by_permutation(csv_results_file_with_static_attr, clf_name):
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
    basin_id_to_std = {}
    for basin_id in basin_id_to_first_ind.keys():
        print(f"in basin: {basin_id}")
        _, _, xs_non_spatial, xs_spatial, _, _ = dataset[basin_id_to_first_ind[basin_id]]
        basin_id_to_std[basin_id] = xs_non_spatial.mean(dim=0).std().item()
    clf = get_clf_from_clf_name(clf_name)
    df_std = pd.DataFrame(basin_id_to_std.items(), columns=["basin_id", "std"])
    df_std["basin_id"] = df_std["basin_id"].astype(int)
    df_results = pd.read_csv(csv_results_file_with_static_attr)
    df_results = df_results.merge(df_std, how='inner', on="basin_id")
    df_results.to_csv("check.csv")
    df_results = process_df_results(df_results)
    create_accumulated_local_effects_and_shap_values(df_results, clf)
    importance = get_feature_importance_from_trained_clf(clf, clf_name, df_results)
    plt.figure(figsize=(25, 20))
    plt.xticks(rotation=90)
    plt.bar([x for x in df_results.columns[:-1]], importance)
    plt.savefig(f"feature_importance_{clf_name}.png")


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
    device = "cpu"
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
    model.eval()
    cam_extractor = SmoothGradCAMpp(model.cnn_lstm.cnn.cnn_layers[4], input_shape=(16, 35, 35))
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
    plot_lon_lat_on_world_map("17775252_17782018_17828539_17832148.csv")
    # create_class_activation_maps_explainable("../checkpoints/TWO_LSTM_CNN_LSTM_epoch_number_30_size_above_1000.pt")
    plt.rc('font', size=12)
    # analyse_results_by_decision_tree("17775252_17782018_17828539_17832148.csv")
    analyse_results_feat_importance_by_permutation("17775252_17782018_17828539_17832148.csv", "logistic_regression")


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    main()
