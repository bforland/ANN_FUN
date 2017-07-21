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
        correct_label = 0
    pass

    outputs = n.query(inputs)

    label = numpy.argmax(outputs)

    if(numpy.max(outputs) > .5):
        if(label == 1):
            print("I'm "+str(round(numpy.max(outputs)*100,2))+"% sure this image contains a car!")
        pass
        if(label == 0):
            print("I'm "+str(round(numpy.max(outputs)*100,2))+"% sure there's no car in this image.")
        pass
        if(label == correct_label):
            print("Correct!")
        else:
            print("Incorrect.")
        pass
    else:
        print("Below confidence threshold. Unable to discriminate.")
    pass
pass
