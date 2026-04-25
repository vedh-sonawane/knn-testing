import numpy as np
import matplotlib.pyplot as plt
from utils import knn_predict

# Dataset
X = np.array([
    [1,2], [2,3], [3,3],
    [6,5], [7,7], [8,6]
])
y = np.array([0,0,0, 1,1,1])

k = 3

# Create grid for decision boundary
x_min, x_max = 0, 10
y_min, y_max = 0, 10

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),
    np.linspace(y_min, y_max, 200)
)

# Predict for every point in grid
Z = []
for i in range(xx.shape[0]):
    row = []
    for j in range(xx.shape[1]):
        point = np.array([xx[i, j], yy[i, j]])
        pred = knn_predict(X, y, point, k)
        row.append(pred)
    Z.append(row)

Z = np.array(Z)

# Plot decision boundary
plt.contourf(xx, yy, Z, alpha=0.3)

# Plot data points
for i in range(len(X)):
    if y[i] == 0:
        plt.scatter(X[i][0], X[i][1])
    else:
        plt.scatter(X[i][0], X[i][1])

plt.title(f"k-NN Decision Boundary (k={k})")
plt.show()