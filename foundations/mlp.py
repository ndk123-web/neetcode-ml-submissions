import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(
        self,
        x: NDArray[np.float64],
        weights: List[NDArray[np.float64]],
        biases: List[NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        total_layers = len(weights)
        res = x

        for i in range(total_layers):

            # weighted sum 
            res = np.dot(res, weights[i]) + biases[i]

            # if its not last layer then add relu
            if i < total_layers - 1:
                res = np.maximum(0, res)

        return np.round(res, 5)
