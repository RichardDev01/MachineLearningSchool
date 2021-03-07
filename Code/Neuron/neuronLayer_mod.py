import neuron_mod as input


class NeuronLayer:
    def __init__(self, inputlist: [input], activationType: str = "Stepper", idLayer: str = "ND"):
        self.inputlist = inputlist
        self.activationType = activationType
        self.idLayer = idLayer

    def activation_triggers(self):
        """
        This function gets all the activation triggers from the layer and return it in a list
        :return:
        """
        outputlist = []
        for perceptron in self.inputlist:
            outputlist.append(perceptron.output)
        return outputlist

    def giveInputs(self, inputvaluelist: [float]):
        """
        Passes the feedforward function through the layers to the neurons
        :param inputvaluelist: a set of inputs for the network
        :return: -
        """
        for perceptron in self.inputlist:
            perceptron.activate(inputvaluelist)

    def getInputString(self):
        """
        This function is for debuging only, it prints al the inputs of the layers
        """
        inputstring = ''
        for inputs in self.inputlist:
            inputstring = inputstring + (str(inputs))
        return inputstring

    def __str__(self):
        return f'\n{self.idLayer} neuron layer has: \n {self.getInputString()} \n ' \
               f'layer triggers{self.activation_triggers()}'
