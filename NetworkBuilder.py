from EvolvingNetwork.Network import Network


def createFlatDenseNetwork() -> Network:
    network = Network(inputSize=28*28, initialHiddenSize=100, outputSize=10)
    network.initialize()
    return network
