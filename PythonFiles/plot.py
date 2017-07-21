import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab as pl

results = numpy.loadtxt("results")

x, y = numpy.unique(results[:,0]), numpy.unique(results[:,1])


X, Y = numpy.meshgrid(x,y)

pl.plot(X,Y,'ro')
Z = numpy.zeros(shape=(len(y)*len(x)))
z_i = 0
for i,j in zip(numpy.ravel(X),numpy.ravel(X)):
    Z[z_i] = results[z_i,3]
    z_i += 1
Z = Z.reshape(X.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z)
plt.show()
