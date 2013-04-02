#!/usr/bin/env python
import numpy as np


def main():
    pass

class MED(object):
    '''array-native MED classifier'''
    def __init__(self, prototypes, **kwargs):
        '''Prototypes should be a dstacked array of vectors'''
        self.prototypes = prototypes

    def classify(self, points):
        Z = points[..., None] - self.prototypes
        Q = np.multiply(Z, Z).sum(axis=1)
        return np.argmin(Q, axis=1)

    def confuse(self, points, labels):
        assert len(points) == labels.size
        k = self.prototypes.shape[2]
        results = self.classify(points)
        return np.bincount(k * labels + results, minlength=k*k).reshape(k, k)


class MICD(object):
    '''array-native MICD classifier'''
    def __init__(self, points, labels):
        invs, means = [], []
        for i in np.unique(labels):
            sel = np.where(labels == i)
            invs.append(np.linalg.inv(np.cov(points[sel].T)))
            means.append(np.mean(points[sel], axis=0))
        self.class_invs = np.dstack(invs)
        self.class_means = np.vstack(means)

    def classify(self, points):
        '''YOLO'''
        Z = points[..., None] - self.class_means.T
        Q = (Z[...,None] * self.class_invs.swapaxes(1, 2)).sum(axis=3)
        return np.argmin((Q * Z).sum(axis=1), axis=1)

    def confuse(self, points, labels):
        assert len(points) == labels.size
        k = self.class_means.shape[0]
        results = self.classify(points)
        return np.bincount(k * labels + results, minlength=k*k).reshape(k, k)


class KMeans(object):
    '''Array-Native KMeans.
    Use 'method' to select which initialization method to use
    options are:
    "Random": [default] randomly choose k-samples from data set as initial
            cluster heads.
    "Forgy":  randomly assign a cluster (from [0, k-1]) to each point
            then compute the cluster mean. Using this as the initial
            cluster head. Note: Unstable.
    '''
    def __init__(self, points, k, method="Random", **kwargs):
        self.k = k
        self.points = points

        if method == 'Forgy':
            labels = np.random.randint(k, size=points.shape[0])
            means = []
            for i in np.arange(k):
                sel = np.where(labels == i)
                means.append(np.mean(points[sel], axis=0))
            means = np.vstack(means)
        elif method == 'Random':
            sel = np.random.choice(np.arange(points.shape[0]),
                                   size=k, replace=False)
            means = points[sel]
        else:
            raise ValueError("Incorrect method")
        self.means = means

    def classify(self, points):
        '''Label some points with their k-means cluster-head'''
        Z = points[..., None] - self.means.T
        Z = np.sqrt((np.abs(Z)**2).sum(axis=1))
        return np.argmin(Z, axis=1)

    def iterate_once(self):
        '''Run a single k-means iteration using the current heads'''
        labels = self.classify(self.points)
        means = []
        for i in np.arange(self.k):  # Looping in python, but less memory
            sel = np.where(labels == i)
            means.append(np.mean(self.points[sel], axis=0))
        return np.vstack(means)

    def execute(self, limit=100):
        '''Run for convergence, will force break if over limit'''
        current = self.means
        for i in xrange(limit):
            self.means = self.iterate_once()
            if np.all(current == self.means):
                return True, i
            current = self.means
        return False, i

if __name__ == '__main__':
    main()
