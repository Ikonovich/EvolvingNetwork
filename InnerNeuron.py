import random
from queue import Queue
from random import randint
from Activations import relu
from EvolvingNetwork.OutputNeuron import OutputNeuron
from EvolvingNetwork.utils import dotProduct, vectorMul, vectorAdd, vectorSub


class InnerNeuron(OutputNeuron):

    def __init__(self, network):
        super().__init__(network)

        # Store the updates received from the input neurons
        self.inputUpdates = list()

        # The list of neurons that this neuron receives input from, in the same order
        # as the weight vector.
        self.inputNeurons = list()

        # Store the forward/input update count. When the size reaches the same as self.inputNeurons,
        # a forward operation is initiated.
        self.inputCounter = 0

        # Store the backprop update count. When the size reaches the same as self.outputNeurons,
        # begins its own backprop operation.
        self.backwardCounter = 0

        # Store the weights wrt to the input neurons
        self.weights = list()
        # Store the bias
        self.bias = 0

        # Store the activation function, which should return both the activation output
        # and the prime (derivative) of the activation output.
        self.activation = relu

        # Stores the last calculated loss.
        self.loss = 0
        # Stores the last calculated delta (loss * output prime).
        self.delta = 0
        # Store the last three states.
        self.history = Queue(maxsize=3)

    # Increments the update counter, and triggers
    # the forward operation when it reaches a particular threshold.
    def update(self):
        self.inputCounter += 1
        if self.inputCounter == len(self.inputNeurons):
            self.inputCounter = 0
            self.forward()

    def backwardUpdate(self, loss):
        self.loss += loss
        self.backwardCounter += 1
        if self.backwardCounter == len(self.outputNeurons):
            self.backwardCounter = 0
            self.backward()

    def backward(self):
        self.delta = self.loss * self.outputPrime

        weightUpdate = vectorMul(self.delta * self.network.learnRate, self.weights)

        self.weights = vectorSub(self.weights, weightUpdate)

        for neuron in self.outputNeurons:
            neuron.backwardUpdate(self.delta * neuron.output)
        self.loss = 0

    # Performs a single iteration of the forward pass.
    # Must always propagate.
    def forward(self):
        # Quick vector multiply.
        value = 0
        for i in range(len(self.inputNeurons)):
            value += self.inputNeurons[i].output * self.weights[i]
        self.output, self.outputPrime = self.activation(value)

        # Propagate the update.
        self.propagate()

    # Adds a single input neuron and the corresponding weight.
    def addInputNeuron(self, neuron):
        self.inputNeurons.append(neuron)
        self.weights.append(random.random() - 0.5)
        neuron.addOutputNeuron(self)

    # Removes a single input neuron and the corresponding weight.
    def removeInputNeuron(self, neuron=None, index: int = None):
        if index is None:
            index = self.inputNeurons.index(neuron)
        # Remove this neuron as an output of the provided neuron.
        self.inputNeurons[index].removeOutputNeuron(index)

        del self.inputNeurons[index]
        del self.weights[index]

