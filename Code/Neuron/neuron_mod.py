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

    def error(self, trainset: ([], [])):
        """
        This error function wil determin the error of a single Neuron
        :param trainset: The training set of the neuron, first the input and second the expected output
        :return: Returns the error in Float value back
        """
        # MSE = Σ | d – y |^2 / n
        error_sum = 0.0
        for index, example in enumerate(trainset):
            # | d – y |^2
            output = self.activate(example[0])

            target = example[1][0]

            error = target - output
            error_sum += error ** 2

        # Σ |error_sum| / n
        error_sum = error_sum / len(trainset)
        return error_sum

    def __str__(self):
        return f'This is a {self.id} Neuron' \
               f' and has {self.inputvaluelist} as input' \
               f' and has {self.inputWeight} as weights' \
               f' and has {self.bias} as bias' \
               f' and Output = {self.output}\n '
