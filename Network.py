import math

from InputNeuron import InputNeuron
from LossFunctions import mse
from Neuron import Neuron
from OutputNeuron import OutputNeuron

from utils import argmax


class Network:

    def __init__(self, input_size, hidden_size, output_size):
        # Store the neuron counts
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Store the non-input neurons contained within this network and their ids.
        self.neurons = dict()
        self.id_to_neuron = dict()
        # Store the highest neuron ID allocated so far.
        self.max_id = 0

        # Store the neurons that receive input directly
        self.inputs = list()

        # Store the output neurons and their output indices
        self.outputs = dict()
        # Store the updates, received from the outputs
        self.updates = None
        # Store the update count. Once this equals the same as the number of output neurons, a result is produced
        # and backprop can start.
        self.update_count = 0

        # Store the final result
        self.result = 0

        # Store the learn rate
        self.learn_rate = 0.05

        # Store the loss function and its results
        self.loss_function = mse
        self.loss = 0
        self.loss_prime = 0

    # Initializes every neuron in the network, in the form of a simple dense linear model.
    def initialize(self):
        # Create the input neurons
        self.createInputNeurons(self.input_size)
        # Create the hidden layer, connecting each neuron to the entire input.
        hidden_connections = [neuron.uuid for neuron in self.inputs]
        output_connections = list()
        for i in range(self.hidden_size):
            uuid = self.createNeuron(hidden_connections)
            output_connections.append(uuid)

        # Create output neurons
        self.updates = [0] * self.output_size
        for i in range(self.output_size):
            uuid = self.createOutputNeuron(output_connections)
            self.outputs[self.id_to_neuron[uuid]] = i

        for neuron in self.id_to_neuron.values():
            neuron.initialize()

    def predict(self, x: list):
        for i in range(len(x)):
            if i > len(self.inputs) - 2:
                print("Pause")

            neuron = self.inputs[i]
            neuron.update(x[i])

        return self.updates

    def backward(self):
        delta = self.loss * self.loss_prime
        for neuron in self.outputs:
            neuron.back_update(delta)

    # Receive an update from an output neuron
    def update(self, neuron):
        index = self.outputs[neuron]
        self.updates[index] = neuron.out

        self.update_count += 1

    def get_loss(self, actual: list, expected: list):
        self.result = argmax(actual)
        self.loss, self.loss_prime = self.loss_function(actual, expected)
        print(f"Prediction is: {self.result} with a loss of {self.loss}")
        return self.result

    # Used for neurons to register with the network
    def register(self, neuron):
        if neuron in self.neurons:
            raise ValueError(f"This neuron is already part of the network, with ID {self.neurons[neuron]}.")
        else:
            # Give the neuron an ID
            self.max_id = self.max_id + 1
            self.neurons[neuron] = self.max_id
            self.id_to_neuron[self.max_id] = neuron
            return self.max_id

    def createInputNeurons(self, size: int):
        for i in range(size):
            neuron = InputNeuron(self)
            self.inputs.append(neuron)

    # Takes a range of neuron IDs
    def createNeuron(self, connection_ids: list[int]) -> int:
        neuron = Neuron(self)

        for entry in connection_ids:
            neuron.add_input(self.id_to_neuron[entry])
        return neuron.uuid

    # Takes a range of neuron IDs
    def createOutputNeuron(self, connection_ids: list[int]) -> int:
        neuron = OutputNeuron(self)

        for entry in connection_ids:
            neuron.add_input(self.id_to_neuron[entry])
        return neuron.uuid