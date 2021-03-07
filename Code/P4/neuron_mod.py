from typing import List

e = 2.71828


def sigmoid(z):
    """
    This is the sigmoid formula used for the sigmoid activation function
    :param z: The Z given in the formula given as a Real number
    :return: Gives back the ouput of the  sigmoid function, a float range from 0 - 1
    """
    return 1 / (1 + e ** -z)


class Neuron:
    def __init__(self, inputWeight: List[float], bias: float = 0, idPerceptron: str = "ND"):
        self.inputWeight = inputWeight
        self.bias = bias
        self.id = idPerceptron
        self.output = 0.0
        self.inputvaluelist = []
        self.total_loss = 0

    def activate(self, inputvaluelist: List[float]):
        """
        This function sets the Perceptron in action. Give a list of inputs for the perceptron the the output wil be
         calculated
        :param inputvaluelist: A Float list with a set of inputs equal to the amounts of weights of this perceptron
        :return: Output of the neuron in float
        """
        if len(inputvaluelist) != len(self.inputWeight):
            raise Exception(f"The length input is {len(inputvaluelist)} and is not equal"
                            f" to length of weights({len(self.inputWeight)})")
        self.inputvaluelist = inputvaluelist
        inputlist = list(zip(inputvaluelist, self.inputWeight))

        input_sum = 0
        for inp in inputlist:
            input_sum += inp[0] * inp[1]
        input_sum += self.bias

        self.output = sigmoid(input_sum)

        return self.output


    def update(self, inputList: [], error, learningRate: float = 0.1):

        #Δwi,j = η ∙ ∂C/∂wi,j = η ∙ outputi ∙ Δj
        # ∂C /∂wi, j = outputi ∙ Δj
        #w'i,j = wi,j – Δwi,j
        for index, weight in enumerate(self.inputWeight):
            self.inputWeight[index] -= learningRate * error * inputList[index]

        # Δbj = η ∙ Δj
        # b'j = bj – Δbj
        self.bias -= learningRate * 1 * error


    def error(self, inputList: [], target: float):
        activation_value = self.activate(inputList)

        # Δj = σ'(inputj) ∙ –(targetj – outputj)
        error = activation_value * (1-activation_value) * -(target-activation_value)

        return error

    def backpropagation(self, inputList: [], target: float, learningRate: float = 0.1):
        # Output neuron error
        #Δj = σ'(inputj) ∙ –(targetj – outputj)
        error = self.error(inputList, target)

        # Hidden neuron error
        # Δi = σ'(inputi) ∙ Σj wi,j ∙ Δj

        self.update(inputList, error, learningRate)

    def calc_total_loss(self, trainset: ([], []), inputList: []):
        sum_difference = 0
        for index, i in enumerate(inputList):
            output = self.activate(list(i))
            difference = (trainset[index][1][0] - output) ** 2
            sum_difference += difference
        self.total_loss = sum_difference/(2*len(inputList))

        return self.total_loss


    def __str__(self):
        return f'This is a {self.id} Neuron' \
               f' |{self.inputvaluelist} as input' \
               f' | {self.inputWeight} as weights' \
               f' | {self.bias} as bias' \
               f' | {self.total_loss} as total loss' \
               f' and Output = {self.output}\n '
