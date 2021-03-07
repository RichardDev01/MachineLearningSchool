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


if __name__ == '__main__':
    sigmoidAND()
    sigmoidXOR()
