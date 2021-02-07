import input_mod as im
import perceptronLayer_mod as ptl
import perceptronNetwork_mod as ptn


def setupNetwork():
    """"Setting up a input array of input for the neural network"""
    inputA = im.InputNW(1, 1)
    inputB = im.InputNW(0.5, 1)
    inputC = im.InputNW(0.5, 0.5)
    inputD = im.InputNW(0, 1)
    inputE = im.InputNW(1, 0.5)

    inputlist = [inputA,inputB,inputC,inputD,inputE]

    layerOne = ptl.perceptronLayer(inputlist)

    networkOne = ptn.PerceptronNetwork( [layerOne])
    testNetwork(networkOne)


def testNetwork(network: ptn.PerceptronNetwork):
    print(network.perceptronLayers[0].inputlist[0], '\n')
    print(network.perceptronLayers[0].getSumInputs(), '\n')
    print(network.perceptronLayers[0].getOutput(), '\n')
    print(network.perceptronLayers[0], '\n')
    print(network.__str__(), '\n')


if __name__ == '__main__':
    setupNetwork()
