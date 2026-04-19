import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, outputs: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        
        output_shifted = outputs - np.max(outputs)  # saves from overflow 

        e_outputs = np.exp(output_shifted)  # find each ones e^z(i)

        # get probability for each one z output
        res = [round(z / np.sum(e_outputs), 4) for z in e_outputs] 

        return res 