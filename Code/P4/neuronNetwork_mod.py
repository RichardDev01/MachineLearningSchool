import neuronLayer_mod as neuronLayer
from typing import List
import time  # time for training
import random  # randomizing training epoch


def devide_list_sum(x):
    """
    This function return a list of combined value on index
    apparently there is a zip function that I never really learned
    this function could be replaced with 'list(zip(*x))'
    [[1,2],[3,4],[5,6] -> [[1,3,5],[2,4,6]]
    @param x: List of sum of values per perceptron from a layer
    @return: List of sum of values for the hidden layer
    """
    lst = []
    for index in range(len(x[0])):
        lst.append([i[index] for i in x])
    return lst


class NeuronNetwork:
    def __init__(self, perceptronLayers: [neuronLayer]):
        self.neuronLayers = perceptronLayers
        self.total_loss = []

    def feed_forward(self, inputvaluelist: List[float]):
        """
        This function feed the input through till the beginning
        @param inputvaluelist: A list of inputs for the input layer
        @return: The output of the network
        """
        value = inputvaluelist
        for layer in self.neuronLayers:
            layer.give_inputs(value)
            value = layer.activation_triggers()

        return value

    def backpropagation_network(self, inputvaluelist: List[float], target: List[float], learningRate: float = 0.1):
        """
        This function uses backpropagation through the network and updates the layers accordingly
        @param inputvaluelist: A list of inputs for the input layer
        @param target: The targets of the network
        @param learningRate: The learningRate of backpropagation
        """
        # Δj = σ'(inputj) ∙ –(targetj – outputj)
        output_error = self.output_layer_difference(inputvaluelist, target)
        for index, layer in enumerate(self.neuronLayers[::-1]):
            if index == 0:
                # Output layer
                #  # Δj = σ'(inputj) ∙ –(targetj – outputj)
                layer.update_neurons(output_error, learningRate)
            else:
                # hidden layers
                # get the values from the previous calculated layer wi,j ∙ Δj
                hidden_sum_of_values = self.neuronLayers[::-1][index - 1].get_hidden_delta()

                # combine the values acordingly
                distrubeted_values = devide_list_sum(hidden_sum_of_values)

                # Δi = σ'(inputi) ∙ Σj
                layer.update_hidden_neurons(distrubeted_values, learningRate)

    def output_layer_difference(self, inputvaluelist: List[float], target: List[float]):
        """
        This function get the difference from the target to the output
        @param inputvaluelist: A list of inputs for the input layer
        @param target: The targets of the network
        @return: a list of output differences
        """
        # activate the network for an output
        output = self.feed_forward(inputvaluelist)

        # Δj = σ'(inputj) ∙ –(targetj – outputj)
        # σ'(inputj) = σ(inputj) ∙ (1 – σ(inputj)) = outputj ∙ (1 – outputj)
        delta_outputs = []
        for index in range(len(target)):
            delta_outputs.append(output[index] * (1-output[index]) * -(target[index] - output[index]))

        return delta_outputs

    def train(self, trainset: ([], []), learningRate: float = 0.1, epochs: int = 1000, time_limit: int = 10):
        """
        This is the main train function what trains the network with backpropagation
        @param trainset: The possible inputs of the network according with the outputs in a set
        @param learningRate: The learning rate of the network
        @param epochs: The amount of training epochs
        @param time_limit: The max time limit of the training session
        """
        start_time = time.time()
        for _ in range(0, epochs):
            # Shuffling the dataset for each training
            random_offset = random.randint(-len(trainset), 0)

            for index, i in enumerate(trainset):
                self.backpropagation_network(trainset[index + random_offset][0], trainset[index + random_offset][1], learningRate)
            current_time = time.time()
            lapsed_time = current_time - start_time

            # Check if the time has surpassed the training time
            if lapsed_time > time_limit:
                break

    def calc_total_loss(self, trainset: ([], [])):
        """
        This function calculates the total loss
        @param trainset: This is the set were we calculate the loss over
        @return: The total lost of the network
        """
        # |di–a(xi)|^2
        sum_difference = []
        for index, i in enumerate(trainset):
            # print(i)
            # output = self.feed_forward(list(i))
            output = self.feed_forward(i[0])
            merged = [(a - b)**2 for a, b in zip(trainset[index][1], output)]
            sum_difference.append(merged)

        summedlist = list(map(sum, zip(*sum_difference)))

        # / 2∙n
        self.total_loss = []
        for elem in summedlist:
            self.total_loss.append(elem/(2*len(trainset)))

        return self.total_loss

    def getlayersinfo(self):
        """
        This function is for debuging only, it prints al the inputs of the layers
        """
        inputstring = ''
        for layer in self.neuronLayers:
            inputstring = inputstring + (str(layer))
        return inputstring

    def __str__(self):
        return f'layers = \\/\n{self.getlayersinfo()}'
