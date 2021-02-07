import input_mod as input
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class perceptronLayer:
    def __init__(self, inputlist: [input]):
        self.inputlist = inputlist

    def setInputs(self, inputs: [input]):
        self.inputlist = input

    def addInput(self, input: input):
        self.inputlist.append(input)

    def getSumInputs(self):
        sumInputs = 0
        for input in self.inputlist:
            sumInputs += input.getValue()
        return sumInputs

    def getOutput(self):
        # return sigmoid(self.getSumInputs)
        return self.getSumInputs

    def __str__(self):
        return f'This p.layer has {self.inputlist} \n ' \
               f'and {self.getSumInputs()} as sum input \n' \
               f' and {self.getOutput()} as layer output '
