#!/usr/bin/env python
from matplotlib import cm
import matplotlib
import matplotlib.pylab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def boundingPrism(A, ax=0):
    '''Return a set of vectors that forms a bounding hyperprism for A'''
    assert len(A.shape) >= ax
    return zip(A.min(ax), A.max(ax))


def plot2d(fn, bounds, dx=0.1):
    '''
    Uses pylab to plot the a scalar function of up to two variables
    argument examples:
    fn = lambda x, y: np.sin(x) + np.cos(x / y + 1)
    bounds = [(-2 * np.pi, 2 * np.pi), (-2, 2)]  # bounding corners of plot
    dx = 0.1  # step size
    '''
    if len(bounds) < 2:
        bounds.append(bounds[0])

    ranges = []
    for lo, hi in bounds[:2]:
        ranges.append(np.arange(lo, hi, dx))

    X, Y = np.meshgrid(*ranges)
    Z = X * 0
    for i in xrange(X.shape[0]):
        for j in xrange(X.shape[1]):
            Z[i, j] = fn(X[i, j], Y[i, j])

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet)
    matplotlib.pylab.show()
