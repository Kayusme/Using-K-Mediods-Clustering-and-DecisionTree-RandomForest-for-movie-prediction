import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from collections import defaultdict
import seaborn as sns
import random

np.set_printoptions(threshold='nan')
pd.set_option('display.max_columns', None)


def load_datas():
    df = pd.read_csv('movie_metadata.csv')
    df = df[['gross', 'imdb_score']].dropna()
    dataset = df.values.tolist()
    cluster = kMedoids(dataset, 5, np.inf, 0)
    print("Clusters=", cluster)


def kMedoids(data, k, prev_cost, count, clusters=None, medoids=None):

    cluster_sum = 0

    while True or count == 100:

        print(count)

        if medoids is None or not medoids:
            medoids = random.sample(data, 5)
        else:
            random.shuffle(medoids)
            medoids.pop()
            medoids.pop()
            medoids.pop()
            medoids+random.sample(data, 3)

        clusters = defaultdict(list)

        for med in medoids:
            temp = []
            for item in data:
                if med is None or not med:
                    break
                else:
                    temp.append(np.linalg.norm(
                        med[0]-item[0])+np.linalg.norm(med[1]-item[1]))

                    min = np.argmin(temp)
                    clusters[min].append(item)

        for i in range(0, len(medoids)):
            inter_cluster = clusters[i]
            for j in range(0, len(inter_cluster)):
                item_cluster = inter_cluster[j]
                medoid = medoids[i]
                cluster_sum += (np.linalg.norm(medoid[0]-item_cluster[0]) +
                                np.linalg.norm(medoid[1]-item_cluster[1]))

        print("Previous Cost= ", prev_cost)
        print("Cluster Sum = ", cluster_sum)

        if cluster_sum < prev_cost:
            prev_cost = cluster_sum
        else:
            break

        count += 1

    return clusters


if __name__ == "__main__":
    load_datas()
