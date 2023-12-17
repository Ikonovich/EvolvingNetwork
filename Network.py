import math

from EvolvingNetwork.BottomNeuron import BottomNeuron
from EvolvingNetwork.InnerNeuron import InnerNeuron
from EvolvingNetwork.TopNeuron import TopNeuron
from LossFunctions import mse
from Neuron import Neuron
from OutputNeuron import OutputNeuron

from utils import argmax


class Network:

    def __init__(self, inputSize: int, initialHiddenSize: int, outputSize: int):
        # Store the top layer neurons - Input/sensory data goes directly into these, one
        # datapoint per neuron.
        self.topNeurons = list()
        # Do inner processing, all initiated by the above neurons.
        self.innerNeurons = list()
        # Formulate output, again initiated by the above neurons.
        self.outputNeurons = list()

        self.inputSize = inputSize
        self.initialHiddenSize = initialHiddenSize
        self.outputSize = outputSize

        # Store the learn rate
        self.learnRate = 0.05

        # Stores the output history.
        self.outputHistory = list()
        # Store the loss function and its results
        self.lossFunction = mse
        self.lossHistory = list()
        self.lossPrime = 0

        # Stores a mapping of neurons to their IDs.
        self.registeredNeurons = dict()
        # Stores the current maximum ID
        self.maxId = 0

    # Initializes every neuron in the network, in the form of a simple dense linear model.
    def initialize(self):
        # Create the input neurons
        self.topNeurons = [TopNeuron(self) for i in range(self.inputSize)]

        # Create the hidden neurons.
        self.innerNeurons = [InnerNeuron(self) for i in range(self.inputSize)]

        # Create the output neurons
        self.outputNeurons = [BottomNeuron(self) for i in range(self.outputSize)]

        # For our top layer, we connect every inner neuron to every top neuron,
        # and every output neuron to every inner neuron.
        # This creates a fully connected one layer perceptron.
        for neuron in self.innerNeurons:
            for topNeuron in self.topNeurons:
                neuron.addInputNeuron(topNeuron)

        for outNeuron in self.outputNeurons:
            for neuron in self.innerNeurons:
                outNeuron.addInputNeuron(neuron)

    def forward(self, x: list[float], train: bool = False):
        if len(x) != len(self.topNeurons):
            raise Exception("Length of network input must match number of top layer neurons.")

        for i in range(len(x)):
            self.topNeurons[i].setOutput(x[i])
        output = [neuron.output for neuron in self.outputNeurons]
        self.outputHistory.append(output)
        return output

    def backward(self):
        # Loss derivative
        delta = self.lossHistory[-1] * self.lossPrime
        for neuron in self.outputNeurons:
            neuron.backwardNetwork(delta * neuron.output)

    def getLoss(self, actual: list[float], expected: list[float]):
        result = argmax(actual)
        loss, self.lossPrime = self.lossFunction(actual, expected)
        self.lossHistory.append(loss)
        print(f"Prediction is: {result} with a loss of {loss}")
        return loss

    # Gets the prediction of a network.
    def getPrediction(self, index: int = None):
        if index is not None:
            return self.outputHistory[index]
        return argmax(self.outputHistory[-1])

    # # Used for neurons to register with the network
    def register(self, neuron):
        if neuron in self.registeredNeurons:
            raise ValueError(f"This neuron is already part of the network, with ID {self.registeredNeurons[neuron]}.")
        else:
            # Give the neuron an ID
            self.maxId = self.maxId + 1
            self.registeredNeurons[neuron] = self.maxId
            return self.maxId
