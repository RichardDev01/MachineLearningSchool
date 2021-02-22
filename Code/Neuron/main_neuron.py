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

from itertools import product # for truth table

def makeTruthTable(length):
    return [p for p in product([1, 0], repeat=length)]


def sigmoidTest():
    AND = im.InputNW(inputWeight=[1, 1, 1], bias=-3, idPerceptron='AND', activationType='Sigmoid')
    AND.activate([1, 1, -1])
    print(f"type = {AND.id} || Input = {AND.inputvaluelist} || Output = {AND.getOutput()}")
    OR = im.InputNW(inputWeight=[1, 1], bias=-1, idPerceptron='OR', activationType='Sigmoid')
    OR.activate([-1, -1])
    print(f"type = {OR.id} || Input = {OR.inputvaluelist} || Output = {OR.getOutput()}")
    OR.activate([-1, 1])
    print(f"type = {OR.id} || Input = {OR.inputvaluelist} || Output = {OR.getOutput()}")
    OR.activate([1, -1])
    print(f"type = {OR.id} || Input = {OR.inputvaluelist} || Output = {OR.getOutput()}")
    OR.activate([1, 1])
    print(f"type = {OR.id} || Input = {OR.inputvaluelist} || Output = {OR.getOutput()}")

    table = makeTruthTable(3)
    NOR = im.InputNW(inputWeight=[-1, -1, -1], bias=0, idPerceptron='NOR', activationType='Sigmoid')

    for i in table:
        NOR.activate(list(i))
        print(f'type = {NOR.id} || input = {list(i)} || output {NOR.getOutput()}')


def sigmoidHalfAdderdefault():
    table = makeTruthTable(2)
    NAND1 = im.InputNW(inputWeight=[0.0, 0.1], bias=0, idPerceptron='NAND 1', activationType='Sigmoid')
    OR1 = im.InputNW(inputWeight=[0.2, 0.3], bias=0, idPerceptron='OR 1', activationType='Sigmoid')
    AND1 = im.InputNW(inputWeight=[0.4, 0.5], bias=0, idPerceptron='AND 1', activationType='Sigmoid')

    AND2 = im.InputNW(inputWeight=[0.6, 0.7, 0.8], bias=0, idPerceptron='AND 2', activationType='Sigmoid')
    REP1 = im.InputNW(inputWeight=[0.9, 1.0, 1.1], bias=0, idPerceptron='REP 1', activationType='Sigmoid')
    # AND2 = im.InputNW(inputWeight=[0.6, 0.7], bias=-2, idPerceptron='AND 2', activationType='Sigmoid')
    # REP1 = im.InputNW(inputWeight=[0, 0], bias=-1, idPerceptron='REP 1', activationType='Sigmoid')

    layer1 = ptl.perceptronLayer([OR1, NAND1, AND1], idLayer='FirstLayer')
    # layer1 = ptl.perceptronLayer([OR1, NAND1], idLayer='FirstLayer')
    layer2 = ptl.perceptronLayer([AND2, REP1], idLayer='SecondLayer')
    # layer2 = ptl.perceptronLayer([AND2], idLayer='SecondLayer')

    # network = ptn.PerceptronNetwork([layer1, layer2])
    network = ptn.PerceptronNetwork([layer1, layer2])
    for i in table:
        output = network.feed_forward(list(i))
        print(f'input = {list(i)} and output {output}')
    print(network, '\n')

def signmoidXor():
    table = makeTruthTable(2)
    NAND1 = im.InputNW(inputWeight=[-1, -1], bias=1, idPerceptron='NAND 1', activationType='Sigmoid')
    OR1 = im.InputNW(inputWeight=[1, 1], bias=-1, idPerceptron='OR 1', activationType='Sigmoid')

    layer1 = ptl.perceptronLayer([OR1, NAND1], idLayer='FirstLayer')

    network = ptn.PerceptronNetwork([layer1])

    for i in table:
        output = network.feed_forward(list(i))
        print(f'input = {list(i)} and output {output}')
    print(network, '\n')


def sigmoidAND():
    expected_output_and = (([1, 1], [True]),
                       ([1, 0], [False]),
                       ([0, 1], [False]),
                       ([0, 0], [False]))
    table = makeTruthTable(2)
    AND1 = im.InputNW(inputWeight=[12, 12], bias=-18, idPerceptron='AND 1', activationType='Sigmoid')

    for i in table:
        # output = network.feed_forward(list(i))
        AND1.activate(list(i))
        output = AND1.getOutput()
        print(f'input = {list(i)} and output {output}')
    print(AND1, '\n' ,AND1.error(expected_output_and))

if __name__ == '__main__':
    """
    This file is purely for debugging, check the notebook
    """
    # feedForwardTest()
    # sigmoidTest()
    # sigmoidHalfAdderdefault()
    # sigmoidHalfAdder()
    # signmoidXor()
    sigmoidAND()
