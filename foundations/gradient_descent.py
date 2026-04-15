class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places

        if iterations == 0:
            return init

        w = float(init)

        for i in range(iterations):

            slope = 2 * w
            w = w - (learning_rate * slope)
        
        return round(w, 5)