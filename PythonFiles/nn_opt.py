import numpy
import pylab as pl
import first_network
import test_nn
import matplotlib as plt
from mpl_toolkits.mplot3d import Axes3D
results = numpy.zeros(shape=(24,4))

row = 0

for i in range(8):

    hidden_nodes_test = (i + 1) * 25;

    for j in range(3):

        learning_rate_test = .1 + (.1 * j)

        for k in range(1):

            epochs_test = 5

            print("Testing NN Number: "+str(row+1))
            print("Parameters (hidden,rate,epochs): "+str(hidden_nodes_test)+", "+str(learning_rate_test)+", "+str(epochs_test))

            n = first_network.initiate(hidden_nodes_test, learning_rate_test)

            first_network.build(n,epochs_test)

            score = test_nn.test(1)

            results[row,0] = hidden_nodes_test
            results[row,1] = learning_rate_test
            results[row,2] = epochs_test
            results[row,3] = score

            row += 1

        pass
    pass
pass

numpy.savetxt("results",results)
