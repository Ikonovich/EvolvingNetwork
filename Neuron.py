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
        # Store the input neurons and their local reference number
        self.inputs = dict()
        # Store the updates received from the input neurons
        self.updates = None
        # Store the fordward update count. When the size reaches the same as self.neurons,
        # a forward operation is initiated.
        self.update_count = 0

        # Store the backward update count. When the size reaches the same as self.neurons,
        # a backward operation is initiated.
        self.back_update_count = 0

        # Store the weights wrt to the input neurons
        self.weights = None
        # Store the bias
        self.bias = 0

        # Store the output neurons - No need for ordering, but list iteration is faster
        self.outputs = list()

        # Store the activation function
        self.activation = relu

        # Store the last calculated output and the derivative
        self.out = 0
        self.out_prime = 0

        # Stores the last calculated loss
        self.loss = 0
        # Store the last three states
        self.history = Queue(maxsize=3)

    # Takes individual inputs from each input neuron.
    # Once all updates have been received, initiates a forward operation.
    def update(self, neuron: "Neuron"):
        index = self.inputs[neuron]
        self.updates[index] = neuron.out
        self.update_count += 1
        if self.update_count == len(self.inputs):
            self.forward()

    # Takes backprop updates inputs from each input neuron.
    # Once all updates have been received, initiates a forward operation.
    def back_update(self, neuron: "Neuron"):
        self.loss += neuron.loss
        self.back_update_count += 1
        if self.back_update_count == len(self.inputs):
            self.backward()

    # Calculate the output and send it to all connected output neurons
    def forward(self):
        self.update_count = 0

        result = 0
        for i in range(len(self.weights)):
            result += self.weights[i] * self.updates[i]
        self.out, self.out_prime = self.activation(result)

        for item in self.outputs:
            item.update(self)

    # Performs a backward operation wrt to the calling neuron
    def backward(self):
        self.back_update_count = 0
        # Update the weights
        # This calculation is reused, so only do it once
        lrn_prime = self.out_prime * self.network.learn_rate
        for i in range(len(self.weights)):
            self.weights[i] -= self.updates[i] * lrn_prime

        # Update the bias
        self.bias += lrn_prime

        # Backpropagate
        for neuron in self.inputs:
            neuron.back_update(self)

    def add_input(self, neuron: "Neuron"):
        if neuron in self.inputs:
            raise ValueError(f"Neuron {self.id} is already an input.")
        else:
            self.inputs[neuron] = len(self.inputs)
            neuron.add_output(self)

    def remove_input(self, neuron: "Neuron"):
        if neuron not in self.inputs:
            raise ValueError(f"Neuron {self.id} is missing from inputs.")
        else:
            del self.inputs[neuron]

    def add_output(self, neuron: "Neuron"):
        if neuron in self.outputs:
            raise ValueError(f"Neuron {self.id} is already an output.")
        else:
            self.outputs.append(neuron)

    def remove_output(self, neuron: "Neuron"):
        for i in range(len(self.outputs)):
            if self.outputs[i] is neuron:
                self.outputs.pop(i)
                self.weights.pop(i)
                return
        raise ValueError("Neuron was not found and could not be removed.")

    # Initializes the neuron weights.
    def initialize(self):
        self.weights = list()
        self.updates = [0] * len(self.inputs)
        for i in range(len(self.inputs)):
            self.weights.append((random.random() - 0.5))


