import input_mod as inputs

class InputCollectorNW:
    def __init__(self, inputlist: [inputs]):
        self.inputlist = inputlist


    def getSumInputs(self):
        sumInputs = 0
        for input in self.inputlist:
            sumInputs += input.getValue()
        return sumInputs
