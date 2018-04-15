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


def assign_target(row, clusters):

    x = row['movie_title']

    for i in range(0, len(clusters.keys())):
        data = clusters.get(i)
        for j in range(0, len(data)):
            df = data[j]
            if df[2] == x:
                row['Cluster'] = 'Cluster'+str(i)

    return row


def plot_graph(clusters):
    markers = ['bo', 'go', 'ro', 'b+', 'r+', 'g+']
    for i in range(0, len(clusters.keys())):
        data = clusters.get(i)
        for j in range(0, len(data)):
            df = data[j]
            plt.plot(df[0], df[1], markers[i])
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

    df_before_split = df.copy()

    split = StratifiedShuffleSplit(
        n_splits=1, test_size=0.2, random_state=0)

    for train_index, test_index in split.split(df, df['Cluster']):
        train_set = df.loc[train_index]
        test_set = df.loc[test_index]

    Y_train = train_set.Cluster
    X_train = train_set[train_set.columns.drop(
        'Cluster').drop('index').drop('movie_title')]
    Y_test = test_set.Cluster
    X_test = test_set[test_set.columns.drop(
        'Cluster').drop('index').drop('movie_title')]

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


def init_app():
    df = pd.read_csv('movie_metadata.csv')
    dataset = df[['gross', 'imdb_score', 'movie_title']
                 ].dropna().values.tolist()
    clusters = kMedoids(dataset, 5, np.inf, 0)
    plot_graph(clusters)

    columns = ['num_user_for_reviews', 'budget',
               'content_rating', 'movie_facebook_likes', 'num_critic_for_reviews', 'movie_title']

    df = df[columns].dropna()
    df = df.reset_index()
    df = df.apply(assign_target, args=(clusters,), axis=1)  # for each row
    build_decision_tree(df)


def kMedoids(data, k, prev_cost, count, clusters=None, medoids=None):

    cluster_sum = 0
    random.seed(0)

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
    init_app()
