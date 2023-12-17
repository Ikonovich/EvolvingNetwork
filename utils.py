from typing import Callable

import torchvision
from torch.utils.data import Dataset


# Performs a dot product of two equal length vectors.
def dotProduct(left: list[float] | float, right: list[float]):
    if left is type(float):
        output = sum([left * x for x in right])
    else:
        output = vectorMul(left, right)
        output = vectorSum(output)
    return output


# Performs single-dimension vector multiplication.
def vectorMul(left: list[float] | float, right: list[float]):
    # Handle scalars provided for the left operator.
    if type(left) is float:
        return [left * x for x in right]

    # Handle left-hand vectors.
    if len(left) != len(right):
        raise Exception("Vector multiply input must be equal length vectors, or a scalar and a vector.")
    value = 0
    for i in range(len(left)):
        value += left[i] * right[i]


# Performs single dimension vector addition.
def vectorAdd(left: list[float] | float, right: list[float]):
    # Handle scalars provided for the left operator.
    if type(left) is float:
        return [left + x for x in right]

    if len(left) != len(right):
        raise Exception("Vector add input must be equal length vectors, or a scalar and a vector.")
    return [left[i] + right[i] for i in range(len(left))]

# Performs single dimension vector subtraction.
def vectorSub(left: list[float] | float, right: list[float]):
    # Handle scalars provided for the left operator.
    if type(left) is float:
        return [left - x for x in right]

    if len(left) != len(right):
        raise Exception("Vector add input must be equal length vectors, or a scalar and a vector.")
    return [left[i] - right[i] for i in range(len(left))]


# Sum over a vector.
def vectorSum(vector: list[float]):
    return sum(vector)


# Gets the index of the maximum value in a sequence.
def argmax(values: list) -> int:
    max_val = float('-inf')
    max_arg = -1

    for i in range(len(values)):
        if values[i] > max_val:
            max_val = values[i]
            max_arg = i
    return max_arg


# Loads the MNIST dataset in a format suitable to be used by a Network object
# for training.
def load_mnist(train: bool = True, transform: Callable = None) -> Dataset:
    dataset = torchvision.datasets.MNIST(root="data/MNIST",
                                         train=train,
                                         download=True,
                                         transform=transform)
    return dataset
