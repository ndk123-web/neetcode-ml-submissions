import numpy as np
from numpy.typing import NDArray


class Solution:

    def sigmoid(self,z):
        return 1 / (1 + np.exp(-z))
    
    def relu(self,z):
        return np.maximum(0,z)

    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        weighted_sum = np.dot(x, w.T) + b

        if activation.lower() == "sigmoid":
            self.pred = self.sigmoid(weighted_sum)
        elif activation.lower() == "relu":
            self.pred = self.relu(weighted_sum)
        
        return np.round(self.pred, 5)
