from EvolvingNetwork.InnerNeuron import InnerNeuron


# This class is intended to act as the unchanging bottom layer of a network.
class BottomNeuron(InnerNeuron):

    def __init__(self, network):
        super().__init__(network)

    def backwardNetwork(self, loss):
        self.loss = loss
        self.delta = loss * self.outputPrime
        self.backward()

