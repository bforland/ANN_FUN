import numpy

import scipy.special

import matplotlib.pyplot

import sys

import first_network

def test(output_style = 0):

    network_structure = numpy.loadtxt("nn_structure")
    
    n = first_network.initiate(int(network_structure[0]), int(network_structure[1]))

    n.wih = numpy.loadtxt("wih")

    n.who = numpy.loadtxt("who")

    # test the neural network

    # load the mnist test data CSV file into a list
    test_data_file = open("mnist_dataset/mnist_test.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    #scorecard for how well the network performs, initially empty
    scorecard = []

    # go through all the records in the test data set
    for record in test_data_list:
        #split the record by the ',' commas
        all_values = record.split(',')
        # correct answer is first value
        correct_label = int(all_values[0])
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
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
