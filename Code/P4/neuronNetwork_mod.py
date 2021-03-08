import neuronLayer_mod as neuronLayer
from typing import List


class NeuronNetwork:
    def __init__(self, perceptronLayers: [neuronLayer]):
        self.neuronLayers = perceptronLayers

    def feed_forward(self, inputvaluelist: List[float]):
        """
        This function feed the input through till the beginning
        """
        value = inputvaluelist
        for layer in self.neuronLayers:
            layer.giveInputs(value)
            value = layer.activation_triggers()

        return value

    def determinLayerTypes(self):
        for layer in self.neuronLayers[:-1]:
            layer.sethiddenlayer()

    def backpropagation_network(self, inputvaluelist: List[float], target: float, learningRate: float = 0.1):
        self.determinLayerTypes()
        for layer in self.neuronLayers:
            layer.determin_errors(inputvaluelist,target,learningRate)


    def getlayersinfo(self):
        """
        This function is for debuging only, it prints al the inputs of the layers
        """
        inputstring = ''
        for layer in self.neuronLayers:
            inputstring = inputstring + (str(layer))
        return inputstring

    def __str__(self):
        return f'layers = \\/\n{self.getlayersinfo()}'
