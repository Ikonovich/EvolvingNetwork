from Activations import sigmoid
from Neuron import Neuron


class OutputNeuron(Neuron):

    def __init__(self, network):
        super().__init__(network)
        # The neurons that receive output from this neuron, and the weight index. Output is propagated
        # in the order of the list.
        self.outputNeurons = list()

        # Store the last calculated output and the derivative
        self.output = 0
        self.outputPrime = 0

    def update(self):
        pass

    def forward(self):
        pass

    # When called, sends an update notification to each of its output neurons.
    def propagate(self):
        for neuron in self.outputNeurons:
            neuron.update()

    # Adds an output neuron to this neuron. Should only be called by the neuron
    # adding this neuron as an input.
    def addOutputNeuron(self, neuron):
        self.outputNeurons.append(neuron)

    # Removes a single output neuron. Should only be called by the neuron
    # removing this neuron as an input.
    def removeOutputNeuron(self, index: int):
        del self.outputNeurons[index]






