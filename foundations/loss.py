import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        
        # protects from log(0) inf
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        res = -np.mean((y_true) * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return np.round(res, 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        
        # y_true = [[1,0,0] ,       [0,1,0]] 
        # y_pred = [[0.4,0.2,0.1] , [0.1, 0.5,0.9]]
        
        # protects from log(0) inf
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

        s = np.sum(y_true * y_pred, axis=1)
        return np.round(-np.mean((np.log(s))), 4)
