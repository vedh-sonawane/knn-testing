import numpy as np

def distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

def knn_predict(X, y, query, k=3):
    distances = []

    for i in range(len(X)):
        d = distance(query, X[i])
        distances.append((d, y[i]))

    distances.sort(key=lambda x: x[0])

    neighbors = distances[:k]
    labels = [label for _, label in neighbors]

    return max(set(labels), key=labels.count)