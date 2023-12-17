import random
from queue import Queue
from random import randint
from Activations import relu


class Neuron:

    def __init__(self, network):
        # Store the network this neuron is part of
        self.network = network
        # Register with the network and get a unique ID
        self.uuid = self.network.register(self)


