import numpy

import scipy.misc

import matplotlib.pyplot

import sys

import first_network

def test(image,label_input):

    network_structure = numpy.loadtxt("nn_structure")

    n = first_network.initiate(int(network_structure[0]), int(network_structure[1]))

    n = first_network.initiate()

    n.wih = numpy.loadtxt("wih")

    n.who = numpy.loadtxt("who")

    # test the neural network

    img_array = scipy.misc.imread(image,flatten=True)
    img_data = 255.0 - img_array.reshape(784)
    img_data = (img_data / 255.0 * .99) + 0.01
    correct_label = int(label_input)

    inputs = img_data

    # query the network
    outputs = n.query(inputs)

    print(outputs)
    # the indes of the highest value corresponds to the label
    label = numpy.argmax(outputs)
    # append correct or incorrect to list
    if(label == correct_label):
        # network's answer matches correct answer, add 1 to scorecard
        print("Correct")
    else:
        # network's answer doesn't match correct answer, add 0 to scorecard
        print("Incorrect")
    pass
pass
