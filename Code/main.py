import perceptron_mod as im
import perceptronLayer_mod as ptl
import perceptronNetwork_mod as ptn


def setupNetwork():
    """"Setting up a input array of input for the neural network"""
    inputA = im.InputNW([[1, 1]], 5)
    inputB = im.InputNW([[0.5, 1]],-8)
    inputC = im.InputNW([[0.5, 0.5]])
    inputD = im.InputNW([[0, 1]])
    inputE = im.InputNW([[1, 0.5]])

    inputA2 = im.InputNW([[1, 1]], 5)
    inputB2 = im.InputNW([[0.5, 1]])
    inputC2 = im.InputNW([[0.5, 0.5]],1)
    inputD2 = im.InputNW([[0, 1]],1)
    inputE2 = im.InputNW([[1, 0.5]],2)

    inputlist = [inputA,inputB,inputC,inputD,inputE]
    inputlist2 = [inputA2, inputB2, inputC2, inputD2, inputE2]

    layerOne = ptl.perceptronLayer(inputlist)
    layerTwo = ptl.perceptronLayer(inputlist2)

    networkOne = ptn.PerceptronNetwork([layerOne, layerTwo])
    testNetwork(networkOne)


def testNetwork(network: ptn.PerceptronNetwork):
    print(f'The fist input \\/\n{network.perceptronLayers[0].inputlist[0]} \n')
    print(f'The sum of inputs of this network is {network.perceptronLayers[0].getSumInputs()}\n')
    print(f'The output of this network is {network.perceptronLayers[0].getOutput()}\n')
    # print(f'The information of this layer is \\/\n{network.perceptronLayers[0]}\n')
    print(f'The information of this layer is \\/\n{network.getlayersInfo()}\n')
    print(f'The forward feed is {network.__str__()}\n')


if __name__ == '__main__':
    setupNetwork()
