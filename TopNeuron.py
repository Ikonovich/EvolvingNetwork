from EvolvingNetwork.OutputNeuron import OutputNeuron


# This class is responsible for taking the input features and propagating them to the rest
# of the network. Each top layer neuron only takes a single datapoint, performs
# no activation, and propagates to any number of lower neurons.
#
# @author evanhnly
class TopNeuron(OutputNeuron):

    def __init__(self, network):
        super().__init__(network)

    # Sets the output of the top layer neuron and causes it to propagate the update.
    def setOutput(self, output: float):
        self.output = output
        self.propagate()
