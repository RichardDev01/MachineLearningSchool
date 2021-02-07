import perceptronLayer_mod as perceptronLayer


class PerceptronNetwork:
    def __init__(self, perceptronLayers: [perceptronLayer]):
        self.perceptronLayers = perceptronLayers

    def feed_forward(self):
        outputlist = []
        for perceptron in self.perceptronLayers:
            outputlist.append(perceptron.getOutput())
        return outputlist

    def __str__(self):
        return f'forwardfeed = {self.feed_forward()} '
