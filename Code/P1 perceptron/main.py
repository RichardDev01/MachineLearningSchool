import perceptron_mod as im
import perceptronLayer_mod as ptl
import perceptronNetwork_mod as ptn


def setupNetwork():
    """"Setting up a input array of input for the neural network"""
    inputA = im.InputNW(inputvaluelist=[1],inputWeight =[1], bias=5)
    inputB = im.InputNW(inputvaluelist=[0.5], inputWeight =[1],bias=-8)
    inputC = im.InputNW(inputvaluelist=[1],inputWeight =[0.5])
    inputD = im.InputNW(inputvaluelist=[0],inputWeight =[1])
    inputE = im.InputNW(inputvaluelist=[0.5],inputWeight =[1])

    inputA2 = im.InputNW(inputvaluelist=[1],inputWeight =[1], bias=5)
    inputB2 = im.InputNW(inputvaluelist=[1,1],inputWeight =[1,1], bias= -2)
    inputC2 = im.InputNW(inputvaluelist=[0.5],inputWeight =[0.5],bias=1)
    inputD2 = im.InputNW(inputvaluelist=[0],inputWeight =[1],bias=1)
    inputE2 = im.InputNW(inputvaluelist=[1,0],inputWeight =[1,1], bias= -2)

    inputlist = [inputA,inputB,inputC,inputD,inputE]
    inputlist2 = [inputA2, inputB2, inputC2, inputD2, inputE2]

    layerOne = ptl.perceptronLayer(inputlist)
    layerTwo = ptl.perceptronLayer(inputlist2)

    networkOne = ptn.PerceptronNetwork([layerOne, layerTwo])
    testNetwork(networkOne)


def testNetwork(network: ptn.PerceptronNetwork):
    # print(f'The fist input \\/\n{network.perceptronLayers[0].inputlist[0]} \n')
    # print(f'The sum of inputs of this network is {network.perceptronLayers[0].getSumInputs()}\n')
    # print(f'The output of this network is {network.perceptronLayers[0]}\n')
    # print(f'The information of this layer is \\/\n{network.perceptronLayers[0]}\n')
    print(f'The information of this layer is \\/\n{network.getlayersInfo()}\n')
    print(f'The forward feed is {network.__str__()}\n')


if __name__ == '__main__':
    setupNetwork()
