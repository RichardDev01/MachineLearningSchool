import neuron_mod as im
import neuronLayer_mod as ptl
import neuronNetwork_mod as ptn


def feedForwardTest():
    NAND1 = im.InputNW(inputWeight =[-1,-1], bias=1, idPerceptron='NAND 1')
    OR1   = im.InputNW(inputWeight=[1, 1], bias=-1, idPerceptron='OR 1')
    AND1  = im.InputNW(inputWeight =[1,1], bias=-2, idPerceptron='AND 1')

    AND2 = im.InputNW(inputWeight =[1,1,0], bias=-2, idPerceptron='NAND 2')
    REP1 = im.InputNW(inputWeight =[0,0,1], bias=-1, idPerceptron='REP 1')

    layer1 = ptl.perceptronLayer([OR1, NAND1, AND1], idLayer='FirstLayer')
    layer2 = ptl.perceptronLayer([AND2, REP1], idLayer='SecondLayer')

    network = ptn.PerceptronNetwork([layer1,layer2])

    print(network.feed_forward([0, 0]))
    print(network.feed_forward([0, 1]))
    print(network.feed_forward([1, 0]))
    print(network.feed_forward([1, 1]))
    print(network)
    # print(layer1)

def sigmoidTest():
    AND = im.InputNW(inputWeight=[0.3, 0.5], bias=-0.7, idPerceptron='AND', activationType='Sigmoid')
    AND.activate([0,0])
    print(AND.getOutput())


if __name__ == '__main__':
    """
    This file is purely for debugging, check the notebook
    """
    # feedForwardTest()
    sigmoidTest()