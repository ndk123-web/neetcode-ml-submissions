import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(
        self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float
    ) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        weighted_sum = np.dot(x, w) + b
        y_pred = self.sigmoid(weighted_sum)

        loss = 0.5 * ((y_pred - y_true) ** 2)

        dL_by_dW = ((y_pred - y_true) * (y_pred * (1 - y_pred)) * x)
        dL_by_dB = (y_pred - y_true) * (y_pred * (1 - y_pred))

        dL_by_dW = np.round(dL_by_dW, 5)
        dL_by_dB = np.round(dL_by_dB, 5)

        return (dL_by_dW, dL_by_dB)
