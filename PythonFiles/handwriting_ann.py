import spicy.misc
import sys

def readimage(sys.argv[1]):
    img_array = scipy.misc.imread(image,flatten=True)
    img_array = numpy.asfarray(img_array)
return img_array

def makegrey(img_array):
    grey = numpy.zeros((img_array.shape[0], img_array.shape[1]))
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
pass

def selectcharacter(img_array):
    for rownum in range(len(img_array)):
        if(numpy.rownum)
    pass
pass
