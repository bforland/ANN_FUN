# python notebook for Make Your Own Neural Network
# code for a 3-layer neural network, and code for learning the MNIST dataset

import numpy

import scipy.special

import matplotlib.pyplot

import sys

import scipy.misc

import glob

# matplotlib inline

# neural network class definition
class neuralNetwork:

#initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
  #set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # link weight  matrices, wih and who
        # weights inside the arrays are w_i_j, where link as
        # from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # learning rate
        self.lr = learningrate

        # activation function is the sigmoid function
        # The expit function, also known as the logistic function,is defined as
        # expit(x) = 1/(1+exp(-x))
        # one can also use the arcTan function numpy.arctan()
        # scipy.special.expit() will use the more standard function
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    # train the neural network
    def train(self, inputs_list, targets_list):
        #convert input list to 2d arrays
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        #calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calcuate the signals emerging from final output layer
        final_outputs  = self.activation_function(final_inputs)

        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights,
        # recombinedat the hidden layers
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # update the weights for the links between the hidden
        # and ouput layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))

        # update the weights for the links between the
        # input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))

        return (self.wih, self.who)

# query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T

        # calculate dignals into hidden layers
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final outputs layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

# number of inputs, hidden and output nodes
inputs_nodes = 4000
output_nodes = 2

# learning rate


def initiate(a = 200, b = 0.2):
    # create instance of neural network
    global hidden_nodes
    global learning_rate
    hidden_nodes = a
    learning_rate = b
    n = neuralNetwork(inputs_nodes,hidden_nodes,output_nodes,learning_rate)

    return n

def build(n, epochs = 5):


    files = glob.glob("CarData/TrainImages/*.pgm")

    training_data_list = []

    for file in files:

        img_array = scipy.misc.imread(file,flatten=True)
        img_array = numpy.asfarray(img_array)
        size = img_array.shape[0] * img_array.shape[1]
        img_data = 255 - img_array.reshape(size)
        img_data = (img_data / 255 *.99) + 0.01
        if("pos" in file):
            training_data_list.append([1,img_data])
        pass
        if("neg" in file):
            training_data_list.append([0,img_data])
        pass
        #print(img_data)
        #imgplot = matplotlib.pyplot.imshow(img_array)
        #matplotlib.pyplot.show()
    pass

    # train the neural network

    # epochs is the number of times the training data set is used for training

    for e in range(epochs):
        # go through all records in the training data set
        print("Epoch: "+str(e + 1))
        for record in training_data_list:
            # split the record by the ',' commas
            #all_values = record.split(',')
            # scale and shift the inputs
            inputs = record[1:]
            # create the target output values (all 0.01, except the
            # desired label which is 0.99)
            targets = numpy.zeros(output_nodes) + 0.01
            # all_values[0] is the target lavel for this record
            targets[int(record[0])] = 0.99
            weights1, weights2 = n.train(inputs, targets)
            pass
        pass

    network_structure = [hidden_nodes, learning_rate, epochs]
    numpy.savetxt("nn_structure", network_structure)
    numpy.savetxt("wih", weights1)
    numpy.savetxt("who", weights2)

    pass
