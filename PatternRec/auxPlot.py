#!/usr/bin/env python
from matplotlib import cm
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg
import matplotlib
import matplotlib.pylab
import matplotlib.pyplot as plt
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


def rotate2D(A, theta):
    '''Perform a rotation transform of a column vector A by theta in radians'''
    R = np.matrix([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return R * A


def plotEllipse(splot, mean, cov, **kwargs):
    '''Given a plot object, add an ellipse defined by mean, and coariance
    matrices'''
    if 'alpha' in kwargs:
        a = kwargs['alpha']
    if 'color' in kwargs:
        c = kwargs['color']

    v, w = linalg.eigh(cov)
    u = w[0] / linalg.norm(w[0])
    theta = np.double(np.arctan2(u[:,1], u[:,0]))
    theta = 180 * theta / np.pi
    ell = Ellipse(mean, 2 * v[0] ** 0.5, 2 * v[1] ** 0.5, 180 + theta, color=c)
    ell.set_clip_box(splot.bbox)
    ell.set_alpha(a)
    splot.add_artist(ell)


def plotEllipse2(u, major, minor, theta=0, **kwargs):
    '''Plot wrapper surrounding the patches.Ellipse artist
    Theta is measured in radians counter-clockwise from the first-axis
    '''

    expand, alpha = False, 0.1
    if expand in kwargs:
        expand = True
    if 'alpha' in kwargs:
        alpha = kwargs['alpha']

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    E = Ellipse(u, major, minor, theta, **kwargs)
    ax.add_artist(E)
    E.set_alpha(alpha)

    # establish some nice axis limits to show the full ellipse
    # find 'corners' of ellipse (unrotated)
    dims = (major / 2.0, minor / 2.0)
    corners = [np.transpose(np.matrix((u[i] - dims[i], u[i] + dims[i]))) for i in xrange(2)]
    corners = [rotate2D(i, theta) for i in corners]
    corners = boundingPrism(np.matrix(corners))

    if expand:  # expands the plot axes
        corners = [(i[0] - abs(i[0]) * 0.5, i[1] + abs(i[1]) * 0.5) for i in corners]

    ax.set_xlim(*corners[0])
    ax.set_ylim(*corners[1])
    matplotlib.pylab.show()
