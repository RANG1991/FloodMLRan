import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn import tree
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression


def fit_clf_analysis(csv_results_file_with_static_attr, clf):
    df_results = pd.read_csv(csv_results_file_with_static_attr)
    df_results["label"] = np.where(df_results['NSE_CNN_LSTM_135'] > df_results['NSE_LSTM_135'], 1, 0)
    df_results = df_results.drop(columns=['NSE_CNN_LSTM_135', 'NSE_LSTM_135'])
    df_results = df_results.set_index("basin_id")
    df_results = df_results.select_dtypes(include=[np.number]).dropna()
    clf.fit(df_results.to_numpy()[:, :-1], df_results["label"])
    return clf, df_results


def analyse_results_by_decision_tree(csv_results_file_with_static_attr):
    clf = DecisionTreeClassifier(random_state=0, max_depth=3)
    clf, df_results = fit_clf_analysis(csv_results_file_with_static_attr, clf)
    importance = clf.feature_importances_
    plt.figure(figsize=(25, 20))
    plt.xticks(rotation=45)
    plt.bar([x for x in df_results.columns[:-1]], importance)
    plt.savefig("decision_tree_analysis.png")


def analyse_results_by_logistic_regression(csv_results_file_with_static_attr):
    clf = LogisticRegression()
    clf, df_results = fit_clf_analysis(csv_results_file_with_static_attr, clf)
    importance = clf.coef_[0]
    plt.figure(figsize=(25, 20))
    plt.xticks(rotation=45)
    plt.bar([x for x in df_results.columns[:-1]], importance)
    plt.savefig("logistic_regression_analysis.png")


def main():
    analyse_results_by_logistic_regression("7307546_7479540.csv")


if __name__ == "__main__":
    main()
