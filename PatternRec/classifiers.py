#!/usr/bin/env python
from bottleneck import argpartsort  # For KNN
from matplotlib.colors import ListedColormap
from pdfs import Multivariate_Normal
import auxPlot
import numpy as np

class PatternClass(object):
    '''Container for training data for pattern recognition tasks
    currently hardcoded for normal distributions'''

    def __init__(self, name, n, mean, cov, c):
        self.name = name
        self.training = np.random.multivariate_normal(mean, cov, n)
        self.testing = np.random.multivariate_normal(mean, cov, n)
        self.mean = np.transpose(np.matrix(mean))
        self.cov = cov
        self.colorCode = c

    def display(self, splot, **kwargs):
        '''
        Adds the scatter of training points, and the unit-deviation ellipse to
        the plot provided.
        '''
        color = None
        marker = 'o'
        if 'color' in kwargs:
            color = kwargs['color']
        if 'marker' in kwargs:
            marker = kwargs['marker']

        splot.scatter(self.training[:, 0], self.training[:, 1], c=color, marker=marker)
        auxPlot.plotEllipse(splot, self.mean, self.cov, color=color, alpha=0.5)

    def bounds(self, training=True):
        '''Return the corners of the bounding box of the training/testing set'''
        if training:
            return zip(self.training.min(0), self.training.max(0))
        else:
            return zip(self.testing.min(0), self.testing.max(0))


class Supervised_Classifier(object):
    ''''Virtual' Class, holds some logistical things like the training classes'''
    def __init__(self, classes, *args, **kwargs):
        assert len(classes) > 1
        self.classes = classes
        self.orderMap = dict((cls.name, i) for i, cls in enumerate(classes))
        self.colorMap = ListedColormap([cls.colorCode for cls in classes])

    def classify(self, x):
        '''Naive classification, overload this!'''
        return None

    def iterRegion(self, bounds, splot, fn, h, **kwargs):
        '''Execute fn across region, spray result to a splot object:
        fn is passed a column vector'''
        xx, yy = np.meshgrid(np.arange(bounds[0][0], bounds[0][1], h),
                             np.arange(bounds[1][0], bounds[1][1], h))
        Z = np.zeros((len(xx), len(yy)))
        xn, yn = xx.shape
        for ix in xrange(xn):
            for iy in xrange(yn):
                Z[ix, iy] = fn(np.c_[xx[ix, iy], yy[ix, iy]].T)
        Z.reshape(xx.shape)

        mode = None
        if 'mode' in kwargs:
            mode = kwargs['mode']

        if mode is None:
            splot.contour(xx, yy, Z, cmap=self.colorMap, linewidth=1)
        else:
            splot.pcolormesh(xx, yy, Z, cmap=self.colorMap, alpha=0.1)


class MED_Classifier(Supervised_Classifier):
    '''
    Given a bunch (more than two) of PatternClass objects,
    trains a MED classifier. Some methods (such as classify) work with multi-
    dimensional data, however the show_ methods are 2D-only.
    '''

    def __init__(self, classes, *args, **kwargs):
        super(MED_Classifier, self).__init__(classes, *args, **kwargs)
        self.tdata = [(i.name, i.mean) for i in self.classes]
        self.rdata = [(i.name, np.average(i.training, axis=0))
                      for i in self.classes]

    @classmethod
    def distance(cls, x, y):
        '''Returns Euclidian distance between two vectors x, and y'''
        return np.sqrt((x - y).T * (x - y))

    def classify(self, x):
        '''Identifies the class name'''
        distances = [(i[0], MED_Classifier.distance(x, i[1])) for i in self.tdata]
        distances.sort(key=lambda x: x[1])
        # TODO: Check if we're on the boundary (unlikely within double-precision)
        return self.orderMap[distances[0][0]]

    def showRegions(self, bounds, splot, h=0.1, **kwargs):
        '''Draws shaded regions over the bounds given, if mode
        is none, draws the boundaries, otherwise shades in regions
        '''
        super(MED_Classifier, self).iterRegion(bounds, splot, self.classify, h, **kwargs)


class GEM_Classifier(Supervised_Classifier):
    '''Trained classifier using training points of several PatternClass
    objects'''
    def __init__(self, classes, *args, **kwargs):
        assert len(classes) > 1  # More Dirty error checking
        super(GEM_Classifier, self).__init__(classes, *args, **kwargs)
        self.tdata = [(cls.name, cls.mean, np.linalg.inv(cls.cov))
                      for cls in self.classes]

    @classmethod
    def distance(cls, x, mean, inv):
        '''
        Determine whitened euclidean distance between pattern x, and a class
        '''
        return np.sqrt((x - mean).T * inv * (x - mean))

    def classify(self, x):
        '''Identifies the class name'''
        distances = [(k, GEM_Classifier.distance(x, i[1], i[2]))
                     for k, i in enumerate(self.tdata)]
        distances.sort(key=lambda y: y[1])
        return distances[0][0]

    def showRegions(self, bounds, splot, h=0.1, **kwargs):
        '''Draws shaded regions over the bounds given, if mode
        is none, draws the boundaries, otherwise shades in regions
        '''
        super(GEM_Classifier, self).iterRegion(bounds, splot, self.classify, h, **kwargs)


class MAP_Classifier(Supervised_Classifier):
    '''Trained Maximum A Priori classifier using training sets of PatternClass
    objects, assuming a Normal distribution centred at PatternClass.mean, with
    PatternClass.cov'''
    def __init__(self, classes, *args, **kwargs):
        super(MAP_Classifier, self).__init__(classes, *args, **kwargs)
        k = len(classes[0].mean)
        assert all(map(lambda p: len(p.mean) == k, classes))
        priorSum = float(sum((len(cls.training) for cls in self.classes)))
        # (Name, prior, probability)
        self.tdata = [(cls.name, len(cls.training) / priorSum,
                      Multivariate_Normal(k, cls.mean, cls.cov))
                      for cls in self.classes]

    def classify(self, x):
        '''Fit x to one of the classes in the classifier'''
        ps = [(k, i[1] * i[2].evaluate(x)) for k, i in enumerate(self.tdata)]
        ps.sort(key=lambda z: z[1], reverse=True)
        return ps[0][0]

    def showRegions(self, bounds, splot, h=0.1, **kwargs):
        '''Draws shaded regions over the bounds given. If mode is none,
        draws nice shaded regions, otherwise gives contour (boundaries).'''
        super(MAP_Classifier, self).iterRegion(bounds, splot, self.classify, h, **kwargs)


class KNN_Classifier(Supervised_Classifier):
    '''Trains a K-Nearest-Neighbour classifier with a bunch of PatternClasses,
    Key difference from classical KNN: No voting, simply find the k-nearest
    neighbours from EACH class, and compute a class-subset mean-prototype.
    From there we then just use MED'''
    def __init__(self, classes, k, *args, **kwargs):
        super(KNN_Classifier, self).__init__(classes, *args, **kwargs)
        self.k = k

    def classify(self, x):
        '''Fit x to one of the classes in the classifier'''
        # Unleash some of numpy with this one ...
        def kmin(arr, k):
            '''returns the indices of the k-smallest items in the array'''
            return argpartsort(arr, k)[:k]

        distances = [np.apply_along_axis(np.linalg.norm, 1, cls.training - x.T)
                    for cls in self.classes]
        idx = [kmin(i, self.k) for i in distances]
        knnMeans = [np.average(self.classes[i].training[j], 0) for i, j in enumerate(idx)]
        d2 = [(i, np.linalg.norm(x.T - j)) for i, j in enumerate(knnMeans)]
        d2.sort(key=lambda z: z[1])
        return d2[0][0]

    def showRegions(self, bounds, splot, h=-1, **kwargs):
        '''Draws shaded regions over the bounds given. If mode isn't 'contour'
        draws nice shaded regions, otherwise gives the (you guessed it) contour
        '''
        super(KNN_Classifier, self).iterRegion(bounds, splot, self.classify, h, **kwargs)
