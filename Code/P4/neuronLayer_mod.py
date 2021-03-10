import neuron_mod as input
from typing import List

class NeuronLayer:
    def __init__(self, inputlist: [input], isHiddenLayer: bool = False, idLayer: str = "ND"):
        self.inputlist = inputlist
        self.isHiddenLayer = isHiddenLayer
        self.idLayer = idLayer
        self.errorList =[]
        self.weightsright = []
        self.errorright = []
        self.hidden_deltas = []

    def activation_triggers(self):
        """
        This function gets all the activation triggers from the layer and return it in a list
        :return:
        """
        outputlist = []
        for neuron in self.inputlist:
            outputlist.append(neuron.output)
        return outputlist

    def update_neurons(self, errors: List, learningRate: float = 0.1):

        for index, neuron in enumerate(self.inputlist):
            neuron.update(errors[index], learningRate)

    def input_weights(self):
        """
        This function gets all the activation triggers from the layer and return it in a list
        :return:
        """
        weightslist = []
        for neuron in self.inputlist:
            weightslist.append(neuron.inputWeight)
        return weightslist

    def update_hidden_neurons(self, deltas: List, learningRate: float = 0.1):
        #[[1,2],[3,4],[5,6] -> [[1,3,5],[2,4,6]]
        # for index, neuron in enumerate(self.inputlist):
        # print(f"i'm here! {deltas}")
        for index, neuron in enumerate(self.inputlist):
            neuron.update_hidden_neuron(deltas[index], learningRate)

    def get_hidden_delta(self):
        self.hidden_deltas = []
        # print(f"--------{self.hidden_deltas}------------------")
        for neuron in self.inputlist:
            self.hidden_deltas.append(neuron.weighted_delta_left_hidden_layers)
        return self.hidden_deltas

    def geterrors(self):
        self.errorList = []
        for neuron in self.inputlist:
            # print(f"{neuron.errorNeuron} dsijuhfdashjioa")
            self.errorList.append(neuron.errorNeuron)
        return self.errorList

    def giveInputs(self, inputvaluelist: [float]):
        """
        Passes the feedforward function through the layers to the neurons
        :param inputvaluelist: a set of inputs for the network
        :return: -
        """
        for neuron in self.inputlist:
            neuron.activate(inputvaluelist)

    def determin_errors(self, inputvaluelist: List[float], target: float, learningRate: float = 0.1):
        for neuron in self.inputlist:
            neuron.backpropagation(inputvaluelist, target, learningRate)
        self.geterrors()

    def getInputString(self):
        """
        This function is for debuging only, it prints al the inputs of the layers
        """
        inputstring = ''
        for inputs in self.inputlist:
            inputstring = inputstring + (str(inputs))
        return inputstring

    def sethiddenlayer(self):
        self.isHiddenLayer = True

    def __str__(self):
        return f'\n{self.idLayer} neuron layer has: \n {self.getInputString()} \n ' \
               f'{self.isHiddenLayer=} \n' \
               f'{self.errorList=} \n' \
               f'layer triggers{self.activation_triggers()}'
