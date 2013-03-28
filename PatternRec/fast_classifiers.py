#!/usr/bin/env python
import numpy as np


def main():
    pass

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

class MED(object):
    '''array-native MED classifier'''
    def __init__(self, prototypes, **kwargs):
        '''Prototypes should be a dstacked array of vectors'''
        self.prototypes = prototypes

    def classify(self, points):
        Z = points[..., np.newaxis] - self.prototypes
        Q = np.multiply(Z, Z).sum(axis=1)
        return np.argmin(Q, axis=1)

    def confuse(self, points, labels):
        assert len(points) == labels.size
        k = self.prototypes.shape[2]
        results = self.classify(points)
        return np.bincount(k * labels + results, minlength=k*k).reshape(k, k)

if __name__ == '__main__':
    main()
