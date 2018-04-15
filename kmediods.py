import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn import tree
from collections import defaultdict
import random
import pydot
from io import StringIO
import pydotplus

np.set_printoptions(threshold='nan')
pd.set_option('display.max_columns', None)


# create a target var based on IMDB score
"""
 8 <= imdb <= 10 => 'great'
 7 <= imdb < 8 => 'good'
 6 <= imdb < 7 => 'average'
 imdb < 6 => 'bad'
"""


def get_movie_class(row):
    if 8 <= row['imdb_score'] <= 10:
        row['Class'] = 'great'
    elif 7 <= row['imdb_score'] < 8:
        row['Class'] = 'good'
    elif 6 <= row['imdb_score'] < 7:
        row['Class'] = 'average'
    else:
        row['Class'] = 'bad'
    return row


def plot_graph(clusters, count):

    for i in range(0, len(clusters.keys())):
        data = clusters.get(i)
        #lb = 'cluster'+str(i+1)
        for j in range(0, len(data)):
            df = data[j]
            plt.scatter(df[0], df[1], c=i, alpha=0.5)
    plt.xlabel('IMDb Scores')
    plt.ylabel('Gross')
    plt.title('K-medoid clusters')
    plt.legend()
    plt.show()


def print_metrics(y_test, y_pred, threshold=0.5):
    print("Precision", metrics.precision_score(y_test, y_pred > threshold))
    print("Recall", metrics.recall_score(y_test, y_pred > threshold))
    print("F1", metrics.f1_score(y_test, y_pred > threshold))
    print("AUC", metrics.roc_auc_score(y_test, y_pred_lr))


def build_decision_tree(df):

    df = df.dropna()
    df = df.reset_index()
    df = df.apply(get_movie_class, axis=1)  # for each row
    df_before_split = df.copy()
    split = StratifiedShuffleSplit(
        n_splits=1, test_size=0.2, random_state=0)
    for train_index, test_index in split.split(df, df['Class']):
        train_set = df.loc[train_index]
        test_set = df.loc[test_index]
        Y_train = train_set.Class
    X_train = train_set[train_set.columns.drop('Class').drop('index')]
    Y_test = test_set.Class
    X_test = test_set[test_set.columns.drop('Class').drop('index')]

    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train, Y_train)
    print('Accuracy', decision_tree.score(X_test, Y_test))
    # Draw graph
    '''dot_data = StringIO()
    export_graphviz(decision_tree, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True, impurity=False, feature_names=train_set.columns.drop('Class').drop('index'))
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png("dtree.png")'''


def load_datas():
    df = pd.read_csv('movie_metadata.csv')
    df = df[['gross', 'imdb_score']].dropna()
    dataset = df.values.tolist()
    clusters = kMedoids(dataset, 5, np.inf, 0)
    plot_graph(clusters, len(clusters.keys()))

    build_decision_tree(df)


def kMedoids(data, k, prev_cost, count, clusters=None, medoids=None):

    cluster_sum = 0

    while True:

        if medoids is None or not medoids:
            medoids = random.sample(data, k)
        else:
            random.shuffle(medoids)
            for _ in range(0, int(k/2)):
                medoids.pop()
            medoids += random.sample(data, int(k/2))

        clusters = defaultdict(list)

        for item in data:
            temp = []
            for i in range(0, len(medoids)):
                med = medoids[i]
                if med is None or not med:
                    break
                else:
                    temp.append(np.linalg.norm(
                        med[0]-item[0])+np.linalg.norm(med[1]-item[1]))
            min_index = np.argmin(temp)
            clusters[min_index].append(item)

        for i in range(0, len(medoids)):
            inter_cluster = clusters[i]
            for j in range(0, len(inter_cluster)):
                item_cluster = inter_cluster[j]
                medoid = medoids[i]
                cluster_sum += (np.linalg.norm(medoid[0]-item_cluster[0]) +
                                np.linalg.norm(medoid[1]-item_cluster[1]))

        if cluster_sum < prev_cost:
            prev_cost = cluster_sum
        else:
            break

        count += 1

    return clusters


if __name__ == "__main__":
    load_datas()
    # plot_graph()
