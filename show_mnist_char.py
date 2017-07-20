import numpy
import matplotlib.pyplot
import scipy.misc
import sys

data_file = open("mnist_dataset/mnist_test.csv")
data_list = data_file.readlines()
data_file.close()

all_values = data_list[1].split(',')
print(all_values[1])
image_array = numpy.asfarray(all_values[1:]).reshape((28,28))
matplotlib.pyplot.imshow(image_array,cmap='Greys',interpolation='None')
matplotlib.pyplot.show()
