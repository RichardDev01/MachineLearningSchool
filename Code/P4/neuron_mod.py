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
        self.errorNeuron = 0
        self.weighted_delta_left_hidden_layers = []

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

    def update(self, error, learningRate: float = 0.1):
        # Δj
        self.errorNeuron = error

        # w'i,j = wi,j – Δwi,j
        for index, weight in enumerate(self.inputWeight):
            self.inputWeight[index] -= learningRate * error * self.inputvaluelist[index]

        # b'j = bj – Δbj
        self.bias -= learningRate * 1 * error

        # pre Calculat sum for hidden errors Left
        self.errors_left_hidden_layers()

    def update_hidden_neuron(self, weigth_delta: List, learningRate: float = 0.1):
        sum_multi_delta = 0
        for weighted_delta in weigth_delta:
            sum_multi_delta += weighted_delta

        error = self.output * (1 - self.output) * sum_multi_delta
        # print(f"this is a hidden uipdate {error}")
        self.errorNeuron = error
        for index, weight in enumerate(self.inputWeight):
            self.inputWeight[index] -= learningRate * error * self.inputvaluelist[index]
        self.bias -= learningRate * 1 * error
        self.errors_left_hidden_layers()

    def errors_left_hidden_layers(self):
        # Σj wi,j ∙ Δj
        self.weighted_delta_left_hidden_layers = []
        for index, weight in enumerate(self.inputWeight):
            self.weighted_delta_left_hidden_layers.append(weight * self.errorNeuron)
        return self.weighted_delta_left_hidden_layers

    def error(self, inputList: [], target: float):
        activation_value = self.activate(inputList)

        # Δj = σ'(inputj) ∙ –(targetj – outputj)
        error = activation_value * (1-activation_value) * -(target-activation_value)
        self.errorNeuron = error
        return error

    def error_hidden_layer(self, inputList: [], weightsRights: [], errors: []):
        # Hidden neuron error
        # Δi = σ'(inputi) ∙ Σj wi,j ∙ Δj

        # σ'(inputi) = outputj ∙ (1 – outputj)
        activation_value = self.activate(inputList)
        activation_value_dx = activation_value * (1-activation_value)

        # Σj wi,j ∙ Δj
        sum_error = 0
        for index in range(len(weightsRights)):
            error = weightsRights[index] * errors[index]
            sum_error += error

        # Δi = σ'(inputi) ∙ Σj wi,j ∙ Δj
        calculated_error = activation_value_dx * sum_error

        self.errorNeuron = calculated_error

        return calculated_error

    def __str__(self):
        return f'This is a {self.id} Neuron' \
               f' |{self.inputvaluelist} as input' \
               f' | {self.inputWeight} as weights' \
               f' | {self.bias} as bias' \
               f' | {self.errorNeuron} as current error' \
               f' and Output = {self.output}\n '
