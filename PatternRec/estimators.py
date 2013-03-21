#!/usr/bin/env python
import numpy as np
from numpy import  matrix
import pdfs


def main():
    pass

def multivariate_normal(data, dim, **kwargs):
    ''''Return the ML multivariate normal distribution object'''
    u, S = matrix(np.mean(data, axis=0)), matrix(np.cov(data))
    return pdfs.Multivariate_Normal(dim, u, S)

def univariate_exponential(data, **kwargs):
    '''Return the ML estimate for univariate exponential distr'''
    l = 1 / np.mean(data, **kwargs)
    return pdfs.Exponential(l)

def multivariate_uniform(data, **kwargs):
    '''Returns the multivariate uniform distribution object.'''
    b = np.min(data, axis=0)
    b = np.append(b, np.max(data, axis=0))
    return pdfs.Uniform(b)

class KDE(object):
    '''Kernel Density Estimator a la Parzen-Rosenblatt.
    Note: Yes, I'm aware that this is better done in scipy.stats.kde.gaussian_kde,
    but I figure the point is to implement the kde, not just use it.'''
    def __init__(self, training, dim, cov):
        '''Hardcoded for Normal kernel'''
        self.kernels = [pdfs.Multivariate_Normal(dim, x.T, cov) for x in training]

    def evaluate(self, x):
        '''Slow as balls with the GIL, but for a one-liner, its expected'''
        # Stick this in your pipe and smoke it.
        return sum(map(lambda y: y.evaluate(x), self.kernels)) / float(len(self.kernels))

if __name__ == '__main__':
    main()
