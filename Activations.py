import numpy as np


# Calculate relu and the derivative for a single float
def relu(x: float) -> (float, float):
    if x > 0:
        return x, 1
    else:
        return 0, 0


def sigmoid(x: float) -> (float, float):
    if x >= 0:
        result = (1. / (1. + np.exp(-x)))
    else:
        result = (np.exp(x) / (1. + np.exp(x)))

    prime = x * (1.0 - x)

    return result, prime
