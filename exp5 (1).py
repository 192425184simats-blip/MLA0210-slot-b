import pandas as pd
import numpy as np
from collections import Counter

data = pd.read_csv("knn_dataset.csv")

X = data[['X', 'Y']].values
y = data['Class'].values

k = 3

def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

def knn_predict(X, y, query, k):
    distances = []
    for i in range(len(X)):
        dist = euclidean_distance(X[i], query)
        distances.append((dist, y[i]))
    distances.sort(key=lambda x: x[0])
    k_nearest = distances[:k]
    classes = [label for _, label in k_nearest]
    return Counter(classes).most_common(1)[0][0]

query_point = np.array([5, 5])

prediction = knn_predict(X, y, query_point, k)

print("Query Point:", query_point)
print("Predicted Class:", prediction)
