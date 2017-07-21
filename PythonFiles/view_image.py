import matplotlib.pyplot
import matplotlib.image
import scipy.misc
import numpy
import glob
import sys

import second_network

def iscar(image):

    network_structure = numpy.loadtxt("nn_structure")

    n = second_network.initiate(int(network_structure[0]), int(network_structure[1]))

    n = second_network.initiate()

    n.wih = numpy.loadtxt("wih")

    n.who = numpy.loadtxt("who")

    img_array = scipy.misc.imread(image,flatten=True)
    img_array = numpy.asfarray(img_array)
    size = img_array.shape[0] * img_array.shape[1]
    img_data = 255 - img_array.reshape(size)
    img_data = (img_data / 255 *.99) + 0.01
    inputs = img_data

    if("pos" in image):
        correct_label = 1
    pass
    if("neg" in image):
        correct_label
    pass

    outputs = n.query(inputs)

    print(outputs)

    label = numpy.argmax(outputs)
    if(label == correct_label):
        # network's answer matches correct answer, add 1 to scorecard
        print("Correct")
    else:
        # network's answer doesn't match correct answer, add 0 to scorecard
        print("Incorrect")
    pass

pass
