import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(
        self,
        x: List[List[float]],
        gamma: List[float],
        beta: List[float],
        running_mean: List[float],
        running_var: List[float],
        momentum: float,
        eps: float,
        training: bool
    ) -> Tuple[List[List[float]], List[float], List[float]]:

        # convert to numpy arrays
        x = np.array(x, dtype=np.float64)
        gamma = np.array(gamma, dtype=np.float64)
        beta = np.array(beta, dtype=np.float64)
        running_mean = np.array(running_mean, dtype=np.float64)
        running_var = np.array(running_var, dtype=np.float64)

        if training:

            # STEP 1 -> mean of each feature(column)
            batch_mean = np.mean(x, axis=0)

            # STEP 2 -> variance of each feature(column)
            batch_var = np.var(x, axis=0)

            # STEP 3 -> normalize
            x_hat = (x - batch_mean) / np.sqrt(batch_var + eps)

            # STEP 4 -> scale and shift
            out = gamma * x_hat + beta

            # STEP 5 -> update running statistics
            running_mean = (1 - momentum) * running_mean + momentum * batch_mean

            running_var = (1 - momentum) * running_var + momentum * batch_var

        else:

            # inference mode uses running stats
            x_hat = (x - running_mean) / np.sqrt(running_var + eps)

            out = gamma * x_hat + beta

        return (
            np.round(out, 4).tolist(),
            np.round(running_mean, 4).tolist(),
            np.round(running_var, 4).tolist()
        )