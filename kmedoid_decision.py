import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn import tree
from collections import defaultdict
import random
import pydot
from io import StringIO
import pydotplus
from multiprocessing import Process
import tkinter as tk
from tkinter import filedialog
from AppGUI import EditorApp

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


def plot_graph(clusters):
    markers = ['bo', 'go', 'ro', 'c+', 'm+', 'y+']
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


def assign_target(row, clusters):

    x = row['movie_title']

    for i in range(0, len(clusters.keys())):
        data = clusters.get(i)
        for j in range(0, len(data)):
            df = data[j]
            if df[2] == x:
                row['cluster'] = 'cluster'+str(i)

    return row

def open_csv(root):
    root.filename = filedialog.askopenfilename(parent=root,filetypes=[("All files","*.csv")])
    print(root.filename)

def gui(df):

#   start
    root = tk.Tk()
    editor = EditorApp(root, df)
    root.mainloop()  # until closes window

#   re-assign dataframe
    new_df = editor.df

    print("THIS IS THE NEW DATABASE:")
    print(new_df.to_string(index=False))

def init_app():

    #loading dataset
    df = pd.read_csv('movie_metadata.csv').dropna()
    gui(df)

    #choosing features and running kmediods
    dataset = df[['gross', 'imdb_score', 'movie_title']]
    dataset = dataset.values.tolist()
    clusters = kMedoids(dataset, 5, np.inf, 0)

    #Plot Cluster graph
    p = Process(target=plot_graph, args=(clusters,))
    p.start()
    
    #choosing features for decision tree
    columns = ['num_user_for_reviews', 'budget'
                , 'num_critic_for_reviews', 'movie_title','movie_facebook_likes','num_voted_users','duration']
    df = df[columns]
    df = df.reset_index()
    df = df.apply(assign_target, args=(clusters,), axis=1)
    df.drop(labels = ['movie_title'], axis = 1, inplace = True)

    df_before_split = df.copy()

    #creating training and test sets
    splitSet = StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=0)

    for train_index, test_index in splitSet.split(df, df['cluster']):
        train_set = df.loc[train_index]
        test_set = df.loc[test_index]

    Y_train = train_set.cluster
    X_train = train_set[train_set.columns.drop('cluster').drop('index')]
    Y_test = test_set.cluster
    X_test = test_set[test_set.columns.drop('cluster').drop('index')]

    #Creating decision tree 
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train, Y_train)

    predictions = decision_tree.predict(X_test)

    print('Accuracy of the decision tree=', decision_tree.score(X_test, Y_test))

    print(confusion_matrix(Y_test,predictions))
    print('\n')
    print(classification_report(Y_test,predictions))
    print('\n')

    #Applying random forest classifier
    rfc = RandomForestClassifier(n_estimators=2000)
    rfc.fit(X_train, Y_train)
    print('Random Forest Statistics\n')
    rfc_pred = rfc.predict(X_test)
    print(confusion_matrix(Y_test,rfc_pred))
    print('\n')
    print(classification_report(Y_test,rfc_pred))

    #Visualising the decision tree (runs only in Jupyter notebook)
    '''dot_data = StringIO()
    export_graphviz(decision_tree, out_file=dot_data,
                        filled=True, rounded=True,
                        special_characters=True, impurity=False, feature_names=train_set.columns.drop('cluster').drop('index'))
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png("dtree.png")'''


if __name__=='__main__':
    init_app()