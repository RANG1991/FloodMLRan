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
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from FloodML_2_LSTM_CNN_LSTM import TWO_LSTM_CNN_LSTM
import torch


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
        use_only_precip_feature=True,
        run_with_radar_data=False
    )
    return camels_dataset


def create_class_activation_maps_explainable(checkpoint_path):
    model = TWO_LSTM_CNN_LSTM(
        input_dim=28,
        image_height=37, image_width=37,
        hidden_dim=256,
        sequence_length_conv_lstm=185,
        in_cnn_channels=1,
        dropout=0.4,
        num_static_attributes=27,
        num_dynamic_attributes=1,
        use_only_precip_feature=True)
    dataset = create_CAMELS_dataset()
    _, _, xs_non_spatial, xs_spatial, _, _ = dataset[0]
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to("cuda")
    model.eval()
    target_layers = [model.layer4[-1]]
    input_tensor = (xs_non_spatial, xs_spatial)
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    # targets = [ClassifierOutputTarget(281)]
    # grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    # grayscale_cam = grayscale_cam[0, :]
    # visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)


def main():
    create_class_activation_maps_explainable(
        "/sci/labs/efratmorin/ranga/FloodMLRan/checkpoints/TWO_LSTM_CNN_LSTM_epoch_number_30_size_above_1000.pt")
    # plt.rc('font', size=12)
    # analyse_results_by_decision_tree("17476442_17477923.csv")
    # analyse_results_feat_importance_by_logistic_regression("17476442_17477923.csv")
    # analyse_results_feat_importance_by_decision_tree("17476442_17477923.csv")
    # analyse_results_feat_importance_by_random_forest("17476442_17477923.csv")
    # analyse_results_feat_importance_by_permutation("17476442_17477923.csv")


if __name__ == "__main__":
    main()
