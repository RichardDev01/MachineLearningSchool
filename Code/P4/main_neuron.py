import neuron_mod as im
import neuronLayer_mod as ptl
import neuronNetwork_mod as ptn


from itertools import product # for truth table

def makeTruthTable(length):
    return [p for p in product([1, 0], repeat=length)]


def sigmoidAND():
    """
    Sigmoid and van het werkboek
    """
    expected_output_and = (([1, 1], [True]),
                       ([1, 0], [False]),
                       ([0, 1], [False]),
                       ([0, 0], [False]))[::-1]
    print(f"And Gate expectation = {expected_output_and}")
    table = makeTruthTable(2)[::-1]
    AND1 = im.Neuron(inputWeight=[-0.5, 0.5], bias=1.5, idPerceptron='AND 1')


    # #Iteratie 1
    # print("it 1")
    # AND1.calc_total_loss(expected_output_and, table)
    # print(AND1)
    # AND1.backpropagation([0, 0], 0, 1)
    # AND1.calc_total_loss(expected_output_and, table)
    # print(AND1)
    #
    # # Iteratie 2
    # print("it 2")
    # AND1.calc_total_loss(expected_output_and, table)
    # print(AND1)
    # AND1.backpropagation([1, 0], 0, 1)
    # AND1.calc_total_loss(expected_output_and, table)
    # print(AND1)
    #
    # # Iteratie 3
    # print("it 3")
    # AND1.calc_total_loss(expected_output_and, table)
    # print(AND1)
    # AND1.backpropagation([0, 1], 0, 1)
    # AND1.calc_total_loss(expected_output_and, table)
    # print(AND1)
    #
    # # Iteratie 4
    # print("it 4")
    # AND1.calc_total_loss(expected_output_and, table)
    # print(AND1)
    # AND1.backpropagation([1, 1], 1, 1)
    # AND1.calc_total_loss(expected_output_and, table)
    # print(AND1)
    print("before training")
    print(AND1)
    for index, i in enumerate(table):
        output = AND1.activate(list(i))
        print(f'input = {list(i)} and output {output} expected output = {expected_output_and[index][1]} difference == {(expected_output_and[index][1][0] - output)**2}')
    print(f"The error = {AND1.calc_total_loss(expected_output_and,table)} \n")


    for _ in range(0,1000):
        for index, i in enumerate(table):
            # print(f"{index+1} interation")
            AND1.calc_total_loss(expected_output_and, table)
            # print(AND1)
            # print(f"{list(i)=} and {expected_output_and[index][1][0]=}")
            AND1.backpropagation(list(i), expected_output_and[index][1][0], 1)
            AND1.calc_total_loss(expected_output_and, table)
            # print(AND1)
    print("after training 1000 epochs and 0.1 learning rate")
    print(AND1)
    for index, i in enumerate(table):
        output = AND1.activate(list(i))
        print(f'input = {list(i)} and output {output} expected output = {expected_output_and[index][1]} difference == {(expected_output_and[index][1][0] - output)**2}')
    print(f"The error = {AND1.calc_total_loss(expected_output_and,table)} \n")


def sigmoidANDNetwork():
    """
    Sigmoid and van het werkboek
    """
    expected_output_and = (([1, 1], [True, True]),
                       ([1, 0], [False, False]),
                       ([0, 1], [False, False]),
                       ([0, 0], [False, False]))[::-1]
    print(f"And Gate expectation = {expected_output_and}")
    table = makeTruthTable(2)[::-1]
    AND1 = im.Neuron(inputWeight=[-0.5, 0.5], bias=1.5, idPerceptron='AND 1')
    AND2 = im.Neuron(inputWeight=[-1, 1], bias=3, idPerceptron='AND 2')

    layerone = ptl.NeuronLayer([AND1,AND2], idLayer='FirstLayer')
    networkOneAND1 = ptn.NeuronNetwork([layerone])

    print(networkOneAND1)
    print(f"total loss = {networkOneAND1.calc_total_loss(expected_output_and, table)}")
    networkOneAND1.train(table,expected_output_and,1,10000,10)
    print(networkOneAND1)
    print(f"total loss = {networkOneAND1.calc_total_loss(expected_output_and, table)}")


def sigmoidXOR():
    expected_output_xor = (([1, 1], [False]),
                           ([1, 0], [True]),
                           ([0, 1], [True]),
                           ([0, 0], [False]))
    print(f"XOR Gate expectation = {expected_output_xor}")

    table = makeTruthTable(2)
    OR1 = im.Neuron(inputWeight=[0.2, -0.4], bias=0, idPerceptron='OR 1')
    NAND1 = im.Neuron(inputWeight=[0.7, 0.1], bias=0, idPerceptron='NAND 1')

    AND1 = im.Neuron(inputWeight=[0.6, 0.9], bias=0, idPerceptron='AND 1')

    layerOneXOR = ptl.NeuronLayer([NAND1, OR1], idLayer='FirstLayer')
    layerTwoXOR = ptl.NeuronLayer([AND1], idLayer='SecondLayer')

    networkOneXOR = ptn.NeuronNetwork([layerOneXOR, layerTwoXOR])

    print("before training")
    # print(networkOneXOR)
    for index, i in enumerate(table):
        output = networkOneXOR.feed_forward(list(i))
        print(f'input = {list(i)} and output {output} expected output = {expected_output_xor[index][1]} difference == {expected_output_xor[index][1][0] - output[0]}')
    print(f"total loss = {networkOneXOR.calc_total_loss(expected_output_xor, table)}\n\n")
    networkOneXOR.train(table,expected_output_xor,1,10000,10)

    for index, i in enumerate(table):
        output = networkOneXOR.feed_forward(list(i))
        print(f'input = {list(i)} and output {output} expected output = {expected_output_xor[index][1]} difference == {expected_output_xor[index][1][0] - output[0]}')
    print(f"total loss = {networkOneXOR.calc_total_loss(expected_output_xor, table)}\n\n")

def sigmoidHalfAdder():
    expected_output_ha = (([1, 1], [False, True]),
                           ([1, 0], [True, False]),
                           ([0, 1], [True, False]),
                           ([0, 0], [False, False]))
    print(f"Half adder expectation = {expected_output_ha}")

    table = makeTruthTable(2)
    OR1 = im.Neuron(inputWeight=[0.0, 0.1], bias=0, idPerceptron='OR 1')
    NAND1 = im.Neuron(inputWeight=[0.2, 0.3], bias=0, idPerceptron='NAND 1')
    AND1 = im.Neuron(inputWeight=[0.4, 0.5], bias=0, idPerceptron='AND 1')

    AND2 = im.Neuron(inputWeight=[0.6, 0.7, 0.8], bias=0, idPerceptron='AND 2')
    REP1 = im.Neuron(inputWeight=[1.1, 1.2, 1.3], bias=0, idPerceptron='REP 1')


    layer1 = ptl.NeuronLayer([OR1, NAND1, AND1], idLayer='FirstLayer')
    layer2 = ptl.NeuronLayer([AND2, REP1], idLayer='SecondLayer')

    network = ptn.NeuronNetwork([layer1, layer2])

    print("before training")
    for index, i in enumerate(table):
        output = network.feed_forward(list(i))
        print(f'input = {list(i)} and output {output} expected output = {expected_output_ha[index][1]}')

    print(f"total loss = {network.calc_total_loss(expected_output_ha, table)}")

    for _ in range(0, 10000):
        for index, i in enumerate(table):
            network.backpropagation_network(list(i), expected_output_ha[index][1], 1)

    print("\n\nafter training 10000 epochs and 1 learning rate")
    for index, i in enumerate(table):
        output = network.feed_forward(list(i))
        print(f'input = {list(i)} and output {output} expected output = {expected_output_ha[index][1]}')
    print(f"total loss = {network.calc_total_loss(expected_output_ha, table)}")


def sigmoidHalfAdderTrain():
    expected_output_ha = (([1, 1], [False, True]),
                           ([1, 0], [True, False]),
                           ([0, 1], [True, False]),
                           ([0, 0], [False, False]))
    print(f"Half adder expectation = {expected_output_ha}")

    table = makeTruthTable(2)
    OR1 = im.Neuron(inputWeight=[0.0, 0.1], bias=0, idPerceptron='OR 1')
    NAND1 = im.Neuron(inputWeight=[0.2, 0.3], bias=0, idPerceptron='NAND 1')
    AND1 = im.Neuron(inputWeight=[0.4, 0.5], bias=0, idPerceptron='AND 1')

    AND2 = im.Neuron(inputWeight=[0.6, 0.7, 0.8], bias=0, idPerceptron='AND 2')
    REP1 = im.Neuron(inputWeight=[1.1, 1.2, 1.3], bias=0, idPerceptron='REP 1')


    layer1 = ptl.NeuronLayer([OR1, NAND1, AND1], idLayer='FirstLayer')
    layer2 = ptl.NeuronLayer([AND2, REP1], idLayer='SecondLayer')

    network = ptn.NeuronNetwork([layer1, layer2])

    print("before training")
    for index, i in enumerate(table):
        output = network.feed_forward(list(i))
        print(f'input = {list(i)} and output {output} expected output = {expected_output_ha[index][1]}')

    print(f"total loss = {network.calc_total_loss(expected_output_ha, table)}")

    network.train(table,expected_output_ha, 1, 10000, 10)

    print("\n\nafter training 10000 epochs and 1 learning rate")
    for index, i in enumerate(table):
        output = network.feed_forward(list(i))
        print(f'input = {list(i)} and output {output} expected output = {expected_output_ha[index][1]}')
    print(f"total loss = {network.calc_total_loss(expected_output_ha, table)}")
    # print(AND2)

if __name__ == '__main__':
    # sigmoidAND()
    sigmoidANDNetwork()
    sigmoidXOR()
    # sigmoidHalfAdder()
    sigmoidHalfAdderTrain()
