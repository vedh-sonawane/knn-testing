import numpy as np
import matplotlib.pyplot as plt

# Two classes
X = np.array([
    [1,2], [2,3], [3,3],   # Class 0
    [6,5], [7,7], [8,6]    # Class 1
])

y = np.array([0,0,0, 1,1,1])

def distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

def knn_predict(X, y, query, k=3):
    distances = []
    
    # Step 1: compute distance to all points
    for i in range(len(X)):
        d = distance(query, X[i])
        distances.append((d, y[i]))
    
    # Step 2: sort by distance
    distances.sort(key=lambda x: x[0])
    
    # Step 3: take k nearest
    neighbors = distances[:k]
    
    # Step 4: majority vote
    labels = [label for _, label in neighbors]
    
    return max(set(labels), key=labels.count)

query = np.array([3,5])

prediction = knn_predict(X, y, query, k=4)
print("Prediction:", prediction)

# plot data
for i in range(len(X)):
    if y[i] == 0:
        plt.scatter(X[i][0], X[i][1], color='blue')
    else:
        plt.scatter(X[i][0], X[i][1], color='red')

# plot query point
plt.scatter(query[0], query[1], color='green', s=100)

plt.show()