import torch

from EvolvingNetwork.NetworkBuilder import createFlatDenseNetwork
from utils import load_mnist, argmax


# Test the baseline evolving network with the MNIST dataset.
def mnistTest():

    model = createFlatDenseNetwork()
    dataset = load_mnist()

    # Normalize and convert to a list
    data = dataset.data / 255
    data = torch.flatten(input=data, start_dim=1).tolist()
    targets = dataset.targets.tolist()

    y = list()
    for item in targets:
        one_hot = [0] * 10
        one_hot[item] = 1
        y.append(one_hot)

    train = data[:50000]
    train_y = y[:50000]
    test = data[50000:]
    test_y = y[50000:]

    for x, y in zip(train, train_y):
        output = model.forward(x)
        actual = argmax(output)
        expected = argmax(y)
        print(f"Actual: {actual} Expected: {expected}")
        loss = model.getLoss(actual=output, expected=y)
        model.backward()
        # model.backward()


if __name__ == "__main__":
    mnistTest()


