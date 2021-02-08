import perceptron_mod as input


class perceptronLayer:
    def __init__(self, inputlist: [input], activationType: str = "Stepper"):
        self.inputlist = inputlist
        self.activationType = activationType


    def setInputs(self, inputs: [input]):
        self.inputlist = inputs

    def addInput(self, input: input):
        self.inputlist.append(input)

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
        return f'p.layer has: \n {self.getInputString()} \n ' \
               f'layer triggers{self.activation_triggers()} \n\n'
