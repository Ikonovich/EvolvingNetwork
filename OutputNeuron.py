from Activations import sigmoid
from Neuron import Neuron


class OutputNeuron(Neuron):

    def __init__(self, network):
        super().__init__(network)
        self.activation = sigmoid
        self.loss_prime = 0

    # Takes individual inputs from each input neuron.
    # Once all updates have been received, initiates a forward operation.
    def update(self, neuron: "Neuron"):
        index = self.inputs[neuron]
        self.updates[index] = neuron.out
        self.update_count += 1
        if self.update_count == len(self.inputs):
            self.forward()

    def back_update(self, delta: float):
        self.loss = delta 
        self.back_update_count += 1

    # Calculate the output and send it to the network
    def forward(self):
        super().forward()
        self.network.update(self)

    def add_output(self, neuron: Neuron):
        raise ValueError("Output Neurons can not have additional outputs added.")

    def remove_output(self, neuron: Neuron):
        raise ValueError("Output Neurons can not have outputs removed.")





