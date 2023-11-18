from typing import Callable

import torchvision
from torch.utils.data import Dataset


def argmax(values: list) -> int:
    max_val = float('-inf')
    max_arg = -1

    for i in range(len(values)):
        if values[i] > max_val:
            max_val = values[i]
            max_arg = i
    return max_arg


def load_mnist(train: bool = True, transform: Callable = None) -> Dataset:
    dataset = torchvision.datasets.MNIST(root="data/MNIST",
                                         train=train,
                                         download=True,
                                         transform=transform)
    return dataset
