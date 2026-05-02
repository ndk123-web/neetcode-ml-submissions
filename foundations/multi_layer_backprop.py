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
        x = np.array(x)
        W1 = np.array(W1)
        W2 = np.array(W2)
        b1 = np.array(b1)
        b2 = np.array(b2)
        y_true = np.array(y_true)

        # feed forward
        z1 = x @ W1.T + b1  # Hidden Layer 1
        a1 = np.maximum(0, z1)

        z2 = a1 @ W2.T + b2  # Output Layer

        loss = np.mean((z2 - y_true) ** 2)  # Calculate Loss

        # backpropagation
        n = len(y_true) if y_true.ndim > 0 else 1

        error = z2 - y_true
        dL_by_dZ2 = (2 / n) * (error)
        dZ2_by_dW2 = dL_by_dZ2.reshape(-1, 1) @ a1.reshape(1, -1)
        dZ2_by_dB2 = dL_by_dZ2 * 1

        dL_by_da1 = dL_by_dZ2.reshape(1, -1) @ W2
        dL_by_da1 = dL_by_da1.flatten()
        dL_by_dz1 = dL_by_da1 * (z1 > 0).astype(float)
        dZ_by_dW1 = dL_by_dz1.reshape(-1, 1) @ x.reshape(1, -1)
        dZ_by_dB1 = dL_by_dz1 * 1

        return {
            'loss': round(float(loss), 4),
            'dW1': np.round(dZ_by_dW1, 4).tolist(),
            'db1': np.round(dZ_by_dB1, 4).tolist(),
            'dW2': np.round(dZ2_by_dW2, 4).tolist(),
            'db2': np.round(dZ2_by_dB2, 4).tolist(),
        }