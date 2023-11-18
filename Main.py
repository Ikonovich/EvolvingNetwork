import torch

from Network import Network
from utils import load_mnist, argmax


def createFlatDenseNetwork() -> Network:
    network = Network(input_size=28*28, hidden_size=100, output_size=10)
    network.initialize()
    return network


if __name__ == "__main__":
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
        output = model.predict(x)
        result = argmax(output)
        print(f"Actual: {result} Expected: {y}")
        model.get_loss(actual=output, expected=y)
        model.backward()

