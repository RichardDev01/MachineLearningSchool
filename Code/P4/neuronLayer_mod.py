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

    def activation_triggers(self):
        """
        This function gets all the activation triggers from the layer and return it in a list
        :return:
        """
        outputlist = []
        for neuron in self.inputlist:
            outputlist.append(neuron.output)
        return outputlist


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
