import neuron_mod as im
import neuronLayer_mod as ptl
import neuronNetwork_mod as ptn


from itertools import product # for truth table

def makeTruthTable(length):
    return [p for p in product([1, 0], repeat=length)]


def sigmoidAND():
    expected_output_and = (([1, 1], [True]),
                       ([1, 0], [False]),
                       ([0, 1], [False]),
                       ([0, 0], [False]))
    print(f"And Gate expectation = {expected_output_and}")
    table = makeTruthTable(2)
    AND1 = im.InputNW(inputWeight=[12, 12], bias=-18, idPerceptron='AND 1', activationType='Sigmoid')

    for index, i in enumerate(table):
        AND1.activate(list(i))
        output = AND1.getOutput()
        print(f'input = {list(i)} and output {output} expected output = {expected_output_and[index][1]} difference == {expected_output_and[index][1][0] - output}')
    print(f"The error = {AND1.error(expected_output_and)} \n")


def sigmoidINV():
    expected_output_inv = (([1], [False]),
                           ([0], [True]))
    print(f"Inv Gate expectation = {expected_output_inv}")
    INV1 = im.InputNW(inputWeight=[-12], bias=6, idPerceptron='INV 1', activationType='Sigmoid')

    table = makeTruthTable(1)
    for index, i in enumerate(table):
        INV1.activate(list(i))
        output = INV1.getOutput()
        print(f'input = {list(i)} and output {output} expected output = {expected_output_inv[index][1]} difference == {expected_output_inv[index][1][0] - output}')
    print(f"The error = {INV1.error(expected_output_inv)} \n")


def sigmoidOR():
    expected_output_or = (([1, 1], [True]),
                           ([1, 0], [True]),
                           ([0, 1], [True]),
                           ([0, 0], [False]))
    print(f"OR Gate expectation = {expected_output_or}")

    table = makeTruthTable(2)
    OR1 = im.InputNW(inputWeight=[24, 24], bias=-12, idPerceptron='OR 1', activationType='Sigmoid')

    for index, i in enumerate(table):
        OR1.activate(list(i))
        output = OR1.getOutput()
        print(f'input = {list(i)} and output {output} expected output = {expected_output_or[index][1]} difference == {expected_output_or[index][1][0] - output}')
    print(f"The error = {OR1.error(expected_output_or)} \n")


def sigmoidNOR():
    expected_output_nor = (([1, 1, 1], [False]),
                           ([1, 1, 0], [False]),
                           ([1, 0, 1], [False]),
                           ([1, 0, 0], [False]),
                           ([0, 1, 1], [False]),
                           ([0, 1, 0], [False]),
                           ([0, 0, 1], [False]),
                           ([0, 0, 0], [True]))

    print(f"NOR Gate expectation = {expected_output_nor}")
    table = makeTruthTable(3)
    NOR1 = im.InputNW(inputWeight=[-24, -24, -24], bias=12, idPerceptron='NOR 1', activationType='Sigmoid')

    for index, i in enumerate(table):
        NOR1.activate(list(i))
        output = NOR1.getOutput()
        print(f'input = {list(i)} and output {output} expected output = {expected_output_nor[index][1]} difference == {expected_output_nor[index][1][0] - output}')
    print(f"The error = {NOR1.error(expected_output_nor)} \n")


def sigmoidNAND():
    expected_output_nand = (([1, 1], [False]),
                           ([1, 0], [True]),
                           ([0, 1], [True]),
                           ([0, 0], [True]))
    print(f"NAND Gate expectation = {expected_output_nand}")
    table = makeTruthTable(2)
    NAND1 = im.InputNW(inputWeight=[-12, -12], bias=18, idPerceptron='NAND 1', activationType='Sigmoid')

    for index, i in enumerate(table):
        NAND1.activate(list(i))
        output = NAND1.getOutput()
        print(f'input = {list(i)} and output {output} expected output = {expected_output_nand[index][1]} difference == {expected_output_nand[index][1][0] - output}')
    print(f"The error = {NAND1.error(expected_output_nand)} \n")


def sigmoidXOR():
    expected_output_xor = (([1, 1], [False]),
                           ([1, 0], [True]),
                           ([0, 1], [True]),
                           ([0, 0], [False]))
    print(f"XOR Gate expectation = {expected_output_xor}")

    table = makeTruthTable(2)
    OR1 = im.InputNW(inputWeight=[24, 24], bias=-12, idPerceptron='OR 1', activationType='Sigmoid')
    NAND1 = im.InputNW(inputWeight=[-12, -12], bias=18, idPerceptron='NAND 1', activationType='Sigmoid')
    AND1 = im.InputNW(inputWeight=[12, 12], bias=-18, idPerceptron='AND 1', activationType='Sigmoid')

    layerOneXOR = ptl.perceptronLayer([NAND1, OR1], idLayer='FirstLayer')
    layerTwoXOR = ptl.perceptronLayer([AND1], idLayer='SecondLayer')

    networkOneXOR = ptn.PerceptronNetwork([layerOneXOR, layerTwoXOR])


    for index, i in enumerate(table):
        output = networkOneXOR.feed_forward(list(i))
        print(f'input = {list(i)} and output {output} expected output = {expected_output_xor[index][1]} difference == {expected_output_xor[index][1][0] - output[0]}')
    print(f"The error = ... ik heb nog geen functie die de netwerk kwaliteit bepaald maar het is klein :) \n")


def sigmoidHalfAdder():
    expected_output_ha = (([1, 1], [False, True]),
                           ([1, 0], [True, False]),
                           ([0, 1], [True, False]),
                           ([0, 0], [False, False]))
    print(f"Half adder expectation = {expected_output_ha}")

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
    print(f"The error = ... ik heb nog geen functie die de netwerk kwaliteit bepaald maar het is klein :) \n")

def test_gates():
    sigmoidAND()
    sigmoidINV()
    sigmoidOR()
    sigmoidNOR()
    sigmoidNAND()
    sigmoidXOR()
    sigmoidHalfAdder()

def test_neuron():
    AND1 = im.InputNW(inputWeight=[12, 12], bias=-18, idPerceptron='AND 1', activationType='Sigmoid')

if __name__ == '__main__':
    # test_gates()
    test_neuron()