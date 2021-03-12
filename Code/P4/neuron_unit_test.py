import neuron_mod as im
import neuronLayer_mod as ptl
import neuronNetwork_mod as ptn
import unittest
import numpy as np


class MyTestCase(unittest.TestCase):

    def test_one_and_network(self):
        expected_output_and = (([1, 1], [True]),
                               ([1, 0], [False]),
                               ([0, 1], [False]),
                               ([0, 0], [False]))[::-1]

        AND1 = im.Neuron(inputWeight=[-0.5, 0.5], bias=1.5, idNeuron='AND 1')

        layerone = ptl.NeuronLayer([AND1], idLayer='FirstLayer')
        networkAnd = ptn.NeuronNetwork([layerone])

        networkAnd.train(expected_output_and, 1, 10000, 10)
        for index, i in enumerate(expected_output_and):
            output = networkAnd.feed_forward(i[0])
            np.testing.assert_array_almost_equal(output, i[1], 1)

        print(f"AND network total lost = {networkAnd.calc_total_loss(expected_output_and)}")


    def test_two_and_network(self):
        expected_output_and = (([1, 1], [True, True]),
                               ([1, 0], [False, False]),
                               ([0, 1], [False, False]),
                               ([0, 0], [False, False]))[::-1]

        AND1 = im.Neuron(inputWeight=[-0.5, 0.5], bias=1.5, idNeuron='AND 1')
        AND2 = im.Neuron(inputWeight=[-1, 1], bias=3, idNeuron='AND 2')

        layerone = ptl.NeuronLayer([AND1, AND2], idLayer='FirstLayer')
        networkTwoAnds = ptn.NeuronNetwork([layerone])

        networkTwoAnds.train(expected_output_and, 1, 10000, 10)
        for index, i in enumerate(expected_output_and):
            output = networkTwoAnds.feed_forward(i[0])
            np.testing.assert_array_almost_equal(output, i[1], 1)

        print(f"Two AND network total lost = {networkTwoAnds.calc_total_loss(expected_output_and)}")

    def test_xor_network(self):
        expected_output_xor = (([1, 1], [False]),
                               ([1, 0], [True]),
                               ([0, 1], [True]),
                               ([0, 0], [False]))

        OR1 = im.Neuron(inputWeight=[0.2, -0.4], bias=0, idNeuron='OR 1')
        NAND1 = im.Neuron(inputWeight=[0.7, 0.1], bias=0, idNeuron='NAND 1')

        AND1 = im.Neuron(inputWeight=[0.6, 0.9], bias=0, idNeuron='AND 1')

        layerOneXOR = ptl.NeuronLayer([NAND1, OR1], idLayer='FirstLayer')
        layerTwoXOR = ptl.NeuronLayer([AND1], idLayer='SecondLayer')

        networkXOR = ptn.NeuronNetwork([layerOneXOR, layerTwoXOR])

        networkXOR.train(expected_output_xor, 1, 10000, 10)

        for index, i in enumerate(expected_output_xor):
            output = networkXOR.feed_forward(i[0])
            np.testing.assert_array_almost_equal(output, i[1], 1)

        print(f"XOR network total lost = {networkXOR.calc_total_loss(expected_output_xor)}")

    def test_halfadder_network(self):
        expected_output_ha = (([1, 1], [False, True]),
                              ([1, 0], [True, False]),
                              ([0, 1], [True, False]),
                              ([0, 0], [False, False]))

        OR1 = im.Neuron(inputWeight=[0.0, 0.1], bias=0, idNeuron='OR 1')
        NAND1 = im.Neuron(inputWeight=[0.2, 0.3], bias=0, idNeuron='NAND 1')
        AND1 = im.Neuron(inputWeight=[0.4, 0.5], bias=0, idNeuron='AND 1')

        AND2 = im.Neuron(inputWeight=[0.6, 0.7, 0.8], bias=0, idNeuron='AND 2')
        REP1 = im.Neuron(inputWeight=[1.1, 1.2, 1.3], bias=0, idNeuron='REP 1')

        layer1 = ptl.NeuronLayer([OR1, NAND1, AND1], idLayer='FirstLayer')
        layer2 = ptl.NeuronLayer([AND2, REP1], idLayer='SecondLayer')

        network = ptn.NeuronNetwork([layer1, layer2])

        network.train(expected_output_ha, 1, 10000, 10)

        for index, i in enumerate(expected_output_ha):
            output = network.feed_forward(i[0])
            np.testing.assert_array_almost_equal(output, i[1], 1)
        print(f"Half adder total loss = {network.calc_total_loss(expected_output_ha)}")


if __name__ == '__main__':
    unittest.main()
