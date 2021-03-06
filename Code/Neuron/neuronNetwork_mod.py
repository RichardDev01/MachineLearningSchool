import neuronLayer_mod as perceptronLayer
from typing import List


class NeuronNetwork:
    def __init__(self, perceptronLayers: [perceptronLayer]):
        self.perceptronLayers = perceptronLayers

    def feed_forward(self, inputvaluelist: List[float]):
        """
        This function feed the input through till the beginning
        """
        value = inputvaluelist
        for layer in self.perceptronLayers:
            layer.giveInputs(value)
            value = layer.activation_triggers()

        return value

    def getlayersInfo(self):
        """
        This function is for debuging only, it prints al the inputs of the layers
        """
        inputstring = ''
        for layer in self.perceptronLayers:
            inputstring = inputstring + (str(layer))
        return inputstring

    def __str__(self):
        return f'layers = \\/\n{self.getlayersInfo()}'
