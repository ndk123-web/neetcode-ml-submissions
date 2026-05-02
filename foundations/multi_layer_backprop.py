import numpy as np
from typing import List


class Solution:
    def forward_and_backward(
        self,
        x: List[float],
        W1: List[List[float]],
        b1: List[float],
        W2: List[List[float]],
        b2: List[float],
        y_true: List[float],
    ) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)

        layer1_weighted_sum = np.dot(x, np.array(W1).T) + b1
        layer1_y_pred = np.maximum(layer1_weighted_sum, 0)

        y_pred = np.dot(layer1_y_pred, np.array(W2).T) + b2

        loss = np.mean((y_pred - y_true) ** 2)

        n = len(y_pred)

        # dL/dy_pred
        dL_dy = (2 / n) * (y_pred - y_true)

        # ---- Layer 2 (Linear) ----
        dL_dz2 = dL_dy

        dW2 = np.outer(dL_dz2, layer1_y_pred)
        db2 = dL_dz2

        # ---- Backprop to Layer 1 ----
        dL_da1 = np.dot(np.array(W2).T, dL_dz2)

        # ReLU derivative (IMPORTANT: use z1)
        relu_grad = (layer1_weighted_sum > 0).astype(float)

        dL_dz1 = dL_da1 * relu_grad

        # ---- Layer 1 (Linear) ----
        dW1 = np.outer(dL_dz1, x)
        dW1 = np.round(dW1, 4)
        dW1[dW1 == -0.0] = 0.0
        db1 = dL_dz1

        return {
            "loss": round(float(loss), 4),
            "dW1": np.round(dW1, 4).tolist(),
            "db1": np.round(db1, 4).tolist(),
            "dW2": np.round(dW2, 4).tolist(),
            "db2": np.round(db2, 4).tolist(),
        }
