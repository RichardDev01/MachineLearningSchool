from typing import List


class InputNW:
    def __init__(self, inputWeight: List[float], bias: float = 0, threshold: float = 0, activationType: str = "Stepper", idPerceptron: str = "ND"):
        self.inputWeight = inputWeight
        self.bias = bias
        self.threshold = threshold
        self.activationType = activationType  # This can be expanded on to work with the sigmoid function
        self.id = idPerceptron

    def activate(self, inputvaluelist: List[float]):
        """
        This function sets the Perceptron in action. Give a list of inputs for the perceptron the the output wil be
         lculated
        :param inputvaluelist: A Float list with a set of inputs equal to the amounts of weights of this perceptron
        """
        if len(inputvaluelist) != len(self.inputWeight):
            raise Exception(f"The length input is {len(inputvaluelist)} and is not equal"
                            f" to length of weights({len(self.inputWeight)})")
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
        The Threshold is to 0 by default but can be changed by the initializing of the object
        """
        if self.activationType == 'Sigmoid':
            return 0  # nog maken
        elif self.activationType == 'Stepper':
            return self.getValue() >= self.threshold  # Stepper activation as boolean
        else:
            return self.getValue() >= self.threshold  # Stepper activation as boolean

    def update(self, trainset: ([], []), epoch:int = 1, learningRate: float = 0.1):

        for runtime in range(epoch):
            for index, example in enumerate(trainset):
                # output y = f(w ∙ x) met activatiefunctie f en y ∈ {0, 1}
                self.activate(example[0])
                output = self.getOutput()

                target = example[1][0]

                # error e = d – y
                error = target - output

                # Δw = η ∙ e ∙ x
                # w' = w + Δw
                for index, weight in enumerate(self.inputWeight):
                    self.inputWeight[index] += learningRate * error * example[0][index]

                # Δb = η ∙ e
                # b' = b + Δb
                self.bias += learningRate * error


    def error(self, trainset: ([], [])):
    #MSE = Σ | d – y |^2 / n

        errorSum = 0.0
        for index, example in enumerate(trainset):
            # | d – y |^2
            self.activate(example[0])
            output = self.getOutput()

            target = example[1][0]

            error = target - output
            errorSum += error**2
        
        # Σ |errorSum| / n
        errorSum = errorSum / len(trainset)
        return errorSum


    def __str__(self):
        return f'This is a {self.id} Perceptron' \
               f' and has {self.inputvaluelist} as input' \
               f' and has {self.inputWeight} as weights' \
               f' and has {self.bias} as bias' \
               f' and output of this perceptron = {self.getOutput()}\n '
