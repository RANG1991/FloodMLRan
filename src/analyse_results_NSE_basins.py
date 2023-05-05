import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn import tree
from matplotlib import pyplot as plt


def analyse_results(csv_results_file_with_static_attr):
    clf = DecisionTreeClassifier(random_state=0, max_depth=3)
    df_results = pd.read_csv(csv_results_file_with_static_attr)
    df_results["label"] = np.where(df_results['NSE_CNN_LSTM_513'] > df_results['NSE_LSTM_513'], 1, 0)
    df_results = df_results.drop(columns=['NSE_CNN_LSTM_513', 'NSE_LSTM_513'])
    df_results = df_results.set_index("basin_id")
    df_results = df_results.select_dtypes(include=[np.number]).dropna()
    clf.fit(df_results.to_numpy()[:, :-1], df_results["label"])
    fig = plt.figure(figsize=(25, 20))
    tree.plot_tree(clf, feature_names=df_results.columns[:-1],
                   class_names=["0", "1"])
    plt.savefig("decision_tree.png")


def main():
    analyse_results("6758360_6608804.csv")


if __name__ == "__main__":
    main()
