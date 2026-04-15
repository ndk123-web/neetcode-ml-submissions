import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        # arr = [1 if 1 / (1 + np.exp(-z)) > 0.5 else 0 for i in z]
        # return arr 
        arr = 1 / (1 + np.exp(-z))
        arr = [round(i, 5) for i in arr]
        return arr

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        return np.maximum(0, z)
