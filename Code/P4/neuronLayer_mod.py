import neuron_mod as input
from typing import List


class NeuronLayer:
    def __init__(self, inputlist: [input], idLayer: str = "ND"):
        self.inputlist = inputlist
        self.idLayer = idLayer
        self.weightsright = []
        self.errorright = []
        self.hidden_deltas = []

    def activation_triggers(self):
        """
        This function gets all the activation triggers from the layer and return it in a list
        :return: list of al the outputs in the layer
        """
        outputlist = []
        for neuron in self.inputlist:
            outputlist.append(neuron.output)
        return outputlist

    def update_neurons(self, errors: List, learningRate: float = 0.1):
        """
        This function updates the output layer neurons with the given error and learning rate
        @param errors: List of errors sorted on list of neuron
        @param learningRate: The learning rate for updating the neuron
        """
        for index, neuron in enumerate(self.inputlist):
            neuron.update(errors[index], learningRate)

    def update_hidden_neurons(self, deltas: List, learningRate: float = 0.1):
        """
        This function updates the hidden layer neurons with the given sum of values and learning rate
        @param deltas: List of sum of values sorted on list of neuron
        @param learningRate: The learning rate for updating the neuron
        """
        for index, neuron in enumerate(self.inputlist):
            neuron.update_hidden_neuron(deltas[index], learningRate)

    def get_hidden_delta(self):
        """
        This function gets all the calculated sums for the previous hidden layer
        @return: List of sums of pre calculated values used for the previous hiden layer
        """
        self.hidden_deltas = []
        for neuron in self.inputlist:
            self.hidden_deltas.append(neuron.weighted_delta_left_hidden_layers)
        return self.hidden_deltas

    def give_inputs(self, inputvaluelist: [float]):
        """
        Passes the feedforward function through the layers to the neurons
        :param inputvaluelist: a set of inputs for the network
        :return: -
        """
        for neuron in self.inputlist:
            neuron.activate(inputvaluelist)

    def get_input_string(self):
        """
        This function is for debuging only, it prints al the inputs of the layers
        """
        inputstring = ''
        for inputs in self.inputlist:
            inputstring = inputstring + (str(inputs))
        return inputstring

    def __str__(self):
        return f'\n{self.idLayer} neuron layer has: \n {self.get_input_string()} \n ' \
               f'layer triggers{self.activation_triggers()}'
