import input_mod as input


class perceptronLayer:
    def __init__(self, inputlist: [input], activationType: str = ""):
        self.inputlist = inputlist
        self.activationType = activationType

    def setInputs(self, inputs: [input]):
        self.inputlist = inputs

    def addInput(self, input: input):
        self.inputlist.append(input)

    def getSumInputs(self):
        sumInputs = 0
        for input in self.inputlist:
            sumInputs += input.getValue()
        return sumInputs

    def getOutput(self):
        return self.getSumInputs() > 0 # Stepper activation as boolean

    def getInputString(self):
        inputstring = ''
        for inputs in self.inputlist:
            inputstring = inputstring + (str(inputs))
        return inputstring


    def __str__(self):
        return f'This p.layer has: \n {self.getInputString()} \n ' \
               f'and {self.getSumInputs().__str__()} as sum input \n' \
               f' and {self.getOutput()} as layer output '
