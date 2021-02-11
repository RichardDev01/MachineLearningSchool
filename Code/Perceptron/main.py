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


def testIvert(input):
    I_input = input
    inputInverter = im.InputNW(inputWeight=[-1], bias=0.5)
    inputInverter.activate([I_input])
    inputlistInverter = [inputInverter]
    layerOneInv = ptl.perceptronLayer(inputlistInverter, idLayer= 'Inverter')
    networkOne = ptn.PerceptronNetwork([layerOneInv])

    print(networkOne)
    output = networkOne.feed_forward()
    print(output)

def andNetwork(I_input):
    inputAND = im.InputNW(inputWeight =[1,1], bias=-2)
    inputAND.activate(I_input)
    inputlistAND = [inputAND]
    layerOneAND = ptl.perceptronLayer(inputlistAND)
    networkOneAND = ptn.PerceptronNetwork([layerOneAND])
    print(networkOneAND)

def makeANetwork():
    inputAND = im.InputNW(inputWeight =[1,1], bias=-2)
    # inputAND.activate(I_input)
    inputlistAND = [inputAND]
    layerOneAND = ptl.perceptronLayer(inputlistAND)
    networkOneAND = ptn.PerceptronNetwork([layerOneAND])


def NetworkAndOr(I_input):
    inputAndOr1 = im.InputNW(inputWeight =[1,1], bias=-2)
    inputAndOr1.activate(I_input[:2])

    inputAndOr2 = im.InputNW(inputWeight =[1,1], bias=-1)
    inputAndOr2.activate(I_input[2:])

    inputlistAndOr = [inputAndOr1,inputAndOr2]
    layerOneAndOr = ptl.perceptronLayer(inputlistAndOr)

    networkOneAndOr = ptn.PerceptronNetwork([layerOneAndOr])

    print(layerOneAndOr.activation_triggers())
    print(networkOneAndOr.feed_forward())
    return networkOneAndOr


def NetworkAndInvOR(I_input):
    inputAndInvOR1 = im.InputNW(inputWeight=[1, 1], bias=-2)
    inputAndInvOR1.activate(I_input[:2])

    inputAndInvOR2 = im.InputNW(inputWeight=[-1], bias=0.5)
    inputAndInvOR2.activate(I_input[2:])

    inputlistAndInvOr = [inputAndInvOR1, inputAndInvOR2]
    layerOneAndInvOr = ptl.perceptronLayer(inputlistAndInvOr)

    networkOneAndInvOR = ptn.PerceptronNetwork([layerOneAndInvOr])
    print(networkOneAndInvOR)
    return networkOneAndInvOR


def NetworkXOR(I_input):
    # NAND
    inputXOR1 = im.InputNW(inputWeight=[-1, -1], bias=1)
    inputXOR1.activate(I_input)

    # OR
    inputXOR2 = im.InputNW(inputWeight=[1, 1], bias=-1)
    inputXOR2.activate(I_input)

    inputlistXOR = [inputXOR1, inputXOR2]
    layerOneXOR = ptl.perceptronLayer(inputlistXOR)

    ##############################################

    # AND
    inputXOR3 = im.InputNW(inputWeight=[1, 1], bias=-2)
    inputXOR3.activate(layerOneXOR.activation_triggers())

    inputlist2XOR = [inputXOR3]
    layerTwoXOR = ptl.perceptronLayer(inputlist2XOR)

    networkOneXOR = ptn.PerceptronNetwork([layerTwoXOR])

    return networkOneXOR

if __name__ == '__main__':
    """
    This file is purely for debugging, check the notebook
    """
    # setupNetwork()
    # testIvert(0)
    # testIvert(1)
    # andNetwork([1,0])
    # makeANetwork()
    # NetworkAndOr([0, 0, 0, 0])
    NetworkAndInvOR([0, 0, 0])