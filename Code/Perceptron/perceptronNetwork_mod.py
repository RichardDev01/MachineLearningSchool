import perceptronLayer_mod as perceptronLayer


class PerceptronNetwork:
    def __init__(self, perceptronLayers: [perceptronLayer]):
        self.perceptronLayers = perceptronLayers

    def feed_forward(self):
        """
        Currently, this function is not used corectly and needs to be chacged later
        """
        outputlist = []
        for perceptron in self.perceptronLayers:
            outputlist.append(perceptron.activation_triggers())
        return outputlist

    def getlayersInfo(self):
        """
        This function is for debuging only, it prints al the inputs of the layers
        """
        inputstring = ''
        for layer in self.perceptronLayers:
            inputstring = inputstring + (str(layer))
        return inputstring

    def __str__(self):
        return f'forwardfeed = {self.feed_forward()} \n' \
               f'layers = \\/\n{self.getlayersInfo()}'
