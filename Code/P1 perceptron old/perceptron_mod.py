class InputNW:
    def __init__(self, inputvaluelist: [float], inputWeight: [float], bias: float = 0, activationType: str = "Stepper"):
        # Input list contain as first value the raw input and as 2nd value the weight of that value
        self.inputvaluelist = inputvaluelist
        self.inputWeight = inputWeight
        self.inputlist = list(zip(inputvaluelist, inputWeight))
        self.bias = bias
        self.activationType = activationType
        self.output = self.getOutput()

    def getValue(self):
        input_sum = 0
        for inp in self.inputlist:
            input_sum += inp[0] * inp[1]
        input_sum += self.bias
        return input_sum

    def getOutput(self):
        if self.activationType == 'Sigmoid':
            return 0 # nog maken
        elif self.activationType == 'Stepper': return self.getValue() >= 0 # Stepper activation as boolean
        else: return self.getValue() >= 0 # Stepper activation as boolean

    def __str__(self):
        return f'This input has {self.inputlist} as input' \
               f' and input has {self.bias} as bias\n '
