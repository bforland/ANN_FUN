import matplotlib.pyplot
import matplotlib.image
import matplotlib.cm
import scipy.misc
import numpy
import sys

img_array = scipy.misc.imread(sys.argv[1],flatten=True)
img_array = numpy.asfarray(img_array)
grey = numpy.zeros((img_array.shape[0], img_array.shape[1])) # init 2D numpy array
grey2 = numpy.zeros((img_array.shape[0], img_array.shape[1])) # init 2D numpy array
# get row number
for rownum in range(len(img_array)):
    for colnum in range(len(img_array[rownum])):
        grey[rownum][colnum] = numpy.mean(img_array[rownum][colnum])
        if(numpy.mean(img_array[rownum][colnum]) < numpy.mean(img_array)):
            grey2[rownum][colnum] = numpy.mean(img_array[rownum][colnum])
        else:
            grey2[rownum][colnum] = 255
        pass
    pass
pass
#grey2 = grey - (numpy.ones((img_array.shape[0], img_array.shape[1])) * numpy.mean(grey))
#print(numpy.mean(grey))
#print(numpy.mean(grey2))
matplotlib.pyplot.imshow(grey, cmap = matplotlib.cm.Greys_r)
matplotlib.pyplot.show()
matplotlib.pyplot.imshow(grey2, cmap = matplotlib.cm.Greys_r)
matplotlib.pyplot.show()

scipy.misc.imsave(sys.argv[2],grey)
