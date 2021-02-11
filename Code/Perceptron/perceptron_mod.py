class InputNW:
    def __init__(self, inputWeight: [float], bias: float = 0, threshold: float = 0, activationType: str = "Stepper"):
        self.inputWeight = inputWeight
        self.bias = bias
        self.threshold = threshold
        self.activationType = activationType  # This can be expanded on to work with the sigmoid function

    def activate(self, inputvaluelist: [float]):
        """
        This function sets the Perceptron in action. Give a list of inputs for the perceptron the the output wil be calculated
        :param inputvaluelist: A Float list with a set of inputs equel to the amouts of weights of this perceptron
        """
        self.inputvaluelist = inputvaluelist
        self.inputlist = list(zip(inputvaluelist, self.inputWeight))
        self.output = self.getOutput()

    def getValue(self):
        """
        This function returns the sum of the values given to the perceptron before going through the activation function
        :return: sum of inputs with weights and bias
        """
        input_sum = 0
        for inp in self.inputlist:
            input_sum += inp[0] * inp[1]
        input_sum += self.bias
        return input_sum

    def getOutput(self):
        """
        This function return the output of the perceptron going through the set activation type
        :return:
        """
        if self.activationType == 'Sigmoid':
            return 0  # nog maken
        elif self.activationType == 'Stepper':
            return self.getValue() >= self.threshold  # Stepper activation as boolean
        else:
            return self.getValue() >= self.threshold  # Stepper activation as boolean

    def __str__(self):
        return f'This input has {self.inputlist} as input' \
               f' and has {self.inputWeight} as input' \
               f' and input has {self.bias} as bias\n '
