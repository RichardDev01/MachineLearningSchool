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


def sigmoidAND():
    expected_output_and = (([1, 1], [True]),
                       ([1, 0], [False]),
                       ([0, 1], [False]),
                       ([0, 0], [False]))
    table = makeTruthTable(2)
    AND1 = im.InputNW(inputWeight=[12, 12], bias=-18, idPerceptron='AND 1', activationType='Sigmoid')

    for index, i in enumerate(table):
        # output = network.feed_forward(list(i))
        AND1.activate(list(i))
        output = AND1.getOutput()
        print(f'input = {list(i)} and output {output} expected output = {expected_output_and[index][1]} difference == {expected_output_and[index][1][0] - output}')
    print(AND1, '\n' ,AND1.error(expected_output_and))


def sigmoidINV():
    expected_output_inv = (([1], [False]),
                           ([0], [True]))
    INV1 = im.InputNW(inputWeight=[-12], bias=6, idPerceptron='INV 1', activationType='Sigmoid')

    table = makeTruthTable(1)
    for index, i in enumerate(table):
        # output = network.feed_forward(list(i))
        INV1.activate(list(i))
        output = INV1.getOutput()
        print(f'input = {list(i)} and output {output} expected output = {expected_output_inv[index][1]} difference == {expected_output_inv[index][1][0] - output}')
    print(INV1, '\n' ,INV1.error(expected_output_inv))

def sigmoidOR():
    expected_output_or = (([1, 1], [True]),
                           ([1, 0], [True]),
                           ([0, 1], [True]),
                           ([0, 0], [False]))

    table = makeTruthTable(2)
    OR1 = im.InputNW(inputWeight=[24, 24], bias=-12, idPerceptron='OR 1', activationType='Sigmoid')

    for index, i in enumerate(table):
        # output = network.feed_forward(list(i))
        OR1.activate(list(i))
        output = OR1.getOutput()
        print(f'input = {list(i)} and output {output} expected output = {expected_output_or[index][1]} difference == {expected_output_or[index][1][0] - output}')
    print(OR1, '\n' ,OR1.error(expected_output_or))


def sigmoidNOR():
    expected_output_nor = (([1, 1, 1], [False]),
                           ([1, 1, 0], [False]),
                           ([1, 0, 1], [False]),
                           ([1, 0, 0], [False]),
                           ([0, 1, 1], [False]),
                           ([0, 1, 0], [False]),
                           ([0, 0, 1], [False]),
                           ([0, 0, 0], [True]))

    table = makeTruthTable(3)
    NOR1 = im.InputNW(inputWeight=[-24, -24, -24], bias=12, idPerceptron='NOR 1', activationType='Sigmoid')

    for index, i in enumerate(table):
        # output = network.feed_forward(list(i))
        NOR1.activate(list(i))
        output = NOR1.getOutput()
        print(f'input = {list(i)} and output {output} expected output = {expected_output_nor[index][1]} difference == {expected_output_nor[index][1][0] - output}')
    print(NOR1, '\n' ,NOR1.error(expected_output_nor))


def sigmoidNAND():
    expected_output_nand = (([1, 1], [False]),
                           ([1, 0], [True]),
                           ([0, 1], [True]),
                           ([0, 0], [True]))
    table = makeTruthTable(2)
    NAND1 = im.InputNW(inputWeight=[-12, -12], bias=18, idPerceptron='NAND 1', activationType='Sigmoid')


    for index, i in enumerate(table):
        # output = network.feed_forward(list(i))
        NAND1.activate(list(i))
        output = NAND1.getOutput()
        print(f'input = {list(i)} and output {output} expected output = {expected_output_nand[index][1]} difference == {expected_output_nand[index][1][0] - output}')
    print(NAND1, '\n' ,NAND1.error(expected_output_nand))


def sigmoidXOR():
    expected_output_xor = (([1, 1], [False]),
                           ([1, 0], [True]),
                           ([0, 1], [True]),
                           ([0, 0], [False]))

    table = makeTruthTable(2)
    OR1 = im.InputNW(inputWeight=[24, 24], bias=-12, idPerceptron='OR 1', activationType='Sigmoid')
    NAND1 = im.InputNW(inputWeight=[-12, -12], bias=18, idPerceptron='NAND 1', activationType='Sigmoid')
    AND1 = im.InputNW(inputWeight=[12, 12], bias=-18, idPerceptron='AND 1', activationType='Sigmoid')

    layerOneXOR = ptl.perceptronLayer([NAND1, OR1], idLayer='FirstLayer')
    layerTwoXOR = ptl.perceptronLayer([AND1], idLayer='SecondLayer')

    networkOneXOR = ptn.PerceptronNetwork([layerOneXOR, layerTwoXOR])


    for index, i in enumerate(table):
        # output = network.feed_forward(list(i))
        output = networkOneXOR.feed_forward(list(i))
        print(f'input = {list(i)} and output {output} expected output = {expected_output_xor[index][1]} difference == {expected_output_xor[index][1][0] - output[0]}')
    # print(XOR1, '\n' ,XOR1.error(expected_output_xor))

def sigmoidHalfAdder():
    expected_output_ha = (([1, 1], [False, True]),
                           ([1, 0], [True, False]),
                           ([0, 1], [True, False]),
                           ([0, 0], [False, False]))

    table = makeTruthTable(2)
    OR1 = im.InputNW(inputWeight=[24, 24], bias=-12, idPerceptron='OR 1', activationType='Sigmoid')
    NAND1 = im.InputNW(inputWeight=[-12, -12], bias=18, idPerceptron='NAND 1', activationType='Sigmoid')
    AND1 = im.InputNW(inputWeight=[12, 12], bias=-18, idPerceptron='AND 1', activationType='Sigmoid')

    AND2 = im.InputNW(inputWeight=[12, 12, 0], bias=-18, idPerceptron='AND 2', activationType='Sigmoid')
    REP1 = im.InputNW(inputWeight=[0, 0, 24], bias=-12, idPerceptron='REP 1', activationType='Sigmoid')


    layer1 = ptl.perceptronLayer([OR1, NAND1, AND1], idLayer='FirstLayer')
    layer2 = ptl.perceptronLayer([AND2, REP1], idLayer='SecondLayer')

    network = ptn.PerceptronNetwork([layer1, layer2])
    for index, i in enumerate(table):
        output = network.feed_forward(list(i))
        print(f'input = {list(i)} and output {output} expected output = {expected_output_ha[index][1]} difference == {expected_output_ha[index][1][0] - output[0]}')
    print(network, '\n')


if __name__ == '__main__':
    """
    This file is purely for debugging, check the notebook
    """

    # sigmoidAND()
    # sigmoidINV()
    # sigmoidOR()
    # sigmoidNOR()
    # sigmoidNAND()
    # sigmoidXOR()
    sigmoidHalfAdder()