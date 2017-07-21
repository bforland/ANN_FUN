import numpy

import scipy.special

import matplotlib.pyplot

import sys

import first_network

import glob

import scipy.misc

def test(output_style = 0):

    network_structure = numpy.loadtxt("nn_structure")

    n = first_network.initiate(int(network_structure[0]), int(network_structure[1]))

    n.wih = numpy.loadtxt("wih")

    n.who = numpy.loadtxt("who")

    # test the neural network

    files = glob.glob("CarData/TestImages/*.pgm")

    test_data_list = []

    precard = []

    for file in files:

        img_array = scipy.misc.imread(file,flatten=True)
        img_array = numpy.asfarray(img_array)
        size = img_array.shape[0] * img_array.shape[1]
        img_data = 255 - img_array.reshape(size)
        img_data = (img_data / 255 *.99) + 0.01
        if("pos" in file):
            test_data_list.append([1,img_data])
            precard.append(1)
        pass
        if("neg" in file):
            test_data_list.append([0,img_data])
            precard.append(0)
        pass
        #print(img_data)
        #imgplot = matplotlib.pyplot.imshow(img_array)
        #matplotlib.pyplot.show()
    pass

    #scorecard for how well the network performs, initially empty
    scorecard = []

    # go through all the records in the test data set
    for record in test_data_list:
        #split the record by the ',' commas
        #all_values = record.split(',')
        # correct answer is first value
        correct_label = int(record[0])
        # scale and shift the inputs
        #inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        inputs = record[1:]
        # query the network
        outputs = n.query(inputs)

        # the indes of the highest value corresponds to the label
        label = numpy.argmax(outputs)
        # append correct or incorrect to list
        if(label == correct_label):
            # network's answer matches correct answer, add 1 to scorecard
            scorecard.append(1)
        else:
            # network's answer doesn't match correct answer, add 0 to scorecard
            scorecard.append(0)
        pass
    pass

    # calculate the performance score, the fraction of correct answers
    scorecard_array = numpy.asarray(scorecard)

    if(output_style == 0):

        print ("performance =", scorecard_array.sum() / scorecard_array.size)

    else:

        performance = (scorecard_array.sum() / scorecard_array.size)

        return performance
    pass
pass
