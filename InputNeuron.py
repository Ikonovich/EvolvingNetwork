from Neuron import Neuron


class InputNeuron(Neuron):

    def __init__(self, network):
        super().__init__(network)
        self.data = None

    # Takes individual inputs from each input neuron.
    # Once all updates have been received, initiates a forward operation.
    def update(self, data: float):
        self.data = data
        self.out = data
        self.forward()

    # Calculate the output and send it to all connected output neurons
    def forward(self):
        for item in self.outputs:
            item.update(self)

    def backward(self, neuron: Neuron):
        # Do nothing
        pass

    def add_input(self, neuron: Neuron):
        raise ValueError("Input Neurons can not have additional inputs added.")

    def remove_input(self, neuron: Neuron):
        raise ValueError("Input Neurons can not have inputs removed.")

    # Initializes the neuron weights. For an input neuron, there are no weights.
    def initialize(self):
        pass



