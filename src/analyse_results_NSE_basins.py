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


def main():
    plt.rc('font', size=12)
    analyse_results_by_decision_tree("17476442_17477923.csv")
    analyse_results_feat_importance_by_logistic_regression("17476442_17477923.csv")
    analyse_results_feat_importance_by_decision_tree("17476442_17477923.csv")
    analyse_results_feat_importance_by_random_forest("17476442_17477923.csv")
    analyse_results_feat_importance_by_permutation("17476442_17477923.csv")


if __name__ == "__main__":
    main()
