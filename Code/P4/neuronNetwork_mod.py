import neuronLayer_mod as neuronLayer
from typing import List


def devide_list_sum(x):
    lst = []
    for index in range(len(x[0])):
        lst.append([i[index] for i in x])
    return lst


class NeuronNetwork:
    def __init__(self, perceptronLayers: [neuronLayer]):
        self.neuronLayers = perceptronLayers

    def feed_forward(self, inputvaluelist: List[float]):
        """
        This function feed the input through till the beginning
        """
        value = inputvaluelist
        for layer in self.neuronLayers:
            layer.giveInputs(value)
            value = layer.activation_triggers()

        return value

    def determinLayerTypes(self):
        for layer in self.neuronLayers[:-1]:
            layer.sethiddenlayer()

    def backpropagation_network(self, inputvaluelist: List[float], target: List[float], learningRate: float = 0.1):
        # self.determinLayerTypes()
        # self.neuronLayers[-1].determin_errors(inputvaluelist,target,learningRate)
        # for layer in self.neuronLayers[:-1]:
        #     layer.determin_errors(inputvaluelist,target,learningRate)
        output_error = self.output_layer_difference(inputvaluelist, target)
        for index, layer in enumerate(self.neuronLayers[::-1]):
            if index == 0:
                #Output layer
                layer.update_neurons(output_error, learningRate)
            else:
                #hidden layers
                hidden_deltas = self.neuronLayers[::-1][index - 1].get_hidden_delta()
                # print(hidden_deltas)
                distrubeted_values = devide_list_sum(hidden_deltas)
                # print(distrubeted_values)
                layer.update_hidden_neurons(distrubeted_values, learningRate)

                pass


    def output_layer_difference(self, inputvaluelist: List[float], target: List[float]):
        #Δj = σ'(inputj) ∙ –(targetj – outputj)
        #σ'(inputj) = σ(inputj) ∙ (1 – σ(inputj)) = outputj ∙ (1 – outputj)

        output = self.feed_forward(inputvaluelist)
        delta_outputs = []
        for index in range(len(target)):
            delta_outputs.append(output[index] * (1-output[index]) * -(target[index] - output[index]))
        # print(delta_outputs)
        return delta_outputs

    def train(self):
        pass

    def getlayersinfo(self):
        """
        This function is for debuging only, it prints al the inputs of the layers
        """
        inputstring = ''
        for layer in self.neuronLayers:
            inputstring = inputstring + (str(layer))
        return inputstring


    def calc_total_loss(self, trainset: ([], []), inputList: []):
        sum_difference = []
        for index, i in enumerate(inputList):
            output = self.feed_forward(list(i))

            merged = [a - b for a, b in zip(trainset[index][1], output)]

            sum_difference.append(merged)

        summedlist = list(map(sum, zip(*sum_difference)))
        self.total_loss = []
        for elem in summedlist:
            self.total_loss.append(elem/(2*len(inputList)))

        return self.total_loss

    def __str__(self):
        return f'layers = \\/\n{self.getlayersinfo()}'
