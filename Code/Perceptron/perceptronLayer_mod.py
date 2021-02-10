import perceptron_mod as input


class perceptronLayer:
    def __init__(self, inputlist: [input], activationType: str = "Stepper", idLayer: str = "ND"):
        self.inputlist = inputlist
        self.activationType = activationType
        self.idLayer = idLayer

    def activation_triggers(self):
        outputlist = []
        for perceptron in self.inputlist:
            outputlist.append(perceptron.getOutput())
        return outputlist

    def getInputString(self):
        inputstring = ''
        for inputs in self.inputlist:
            inputstring = inputstring + (str(inputs))
        return inputstring

    def __str__(self):
        return f'{self.idLayer}p.layer has: \n {self.getInputString()} \n ' \
               f'layer triggers{self.activation_triggers()}'
