import numpy as np
from numpy.typing import NDArray
import math

class Solution:
    def forward(self, x: NDArray[np.float64], gamma: NDArray[np.float64], beta: NDArray[np.float64]) -> NDArray[np.float64]:
        
        # find mean,var
        mean = np.mean(x)
        variance = np.mean((x - mean) ** 2)
        epsilon = 1e-5
        
        # find the actual output
        x_hat = ((x - mean) / math.sqrt(variance + epsilon))
        output = x_hat * gamma + beta

        return np.round(output, 5)