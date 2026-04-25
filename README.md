# k-NN from Scratch with Decision Boundary Visualization

## Overview 
This project implements the k-Nearest Neighbors (k-NN) algorithm from scratch and visualizes how it classifies data points by plotting decision boundaries.

---

## What This Project Shows
- How k-NN makes predictions based on nearest neighbors
- How changing **k** affects model behavior
- Visual decision boundaries for classification

---

## Key Concept
k-NN classifies a point based on the majority label of its nearest neighbors.

---

## Visualization

The model generates a grid of points and predicts each one, producing a decision boundary that shows how the model separates classes.

---

## Example

- k = 1 → highly sensitive (overfitting)  
- k = 3 → balanced  
- k = 5 → smoother boundary (underfitting)  

---

## How to Run

```bash
pip install -r requirements.txt
python main.py
