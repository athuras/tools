#!/usr/bin/env python
import numpy as np


class Multivariate_Normal():
    '''
    Multivariate Normal Distribution: N(mu, Sigma)
    Usage: >>N = Gauss.Multivariate(3) # where 3 is dimensionality
           >>N.evaluate([[0],[0],[0]])
           >>0.15915494309189535
    '''

    def __init__(self, k, mu=None, sigma=None, **kwargs):
        '''
        Initialize the distribution with dimensionality,
        and optional means, and/or covariance matrix.
        dataType defaults to np.double
        '''
        dataType = kwargs.get('dataType', np.double)

        if mu is None:
            mu = np.transpose(np.matrix(np.zeros(k, dtype=dataType)))
        elif type(mu) is not np.matrix:
            mu = np.transpose(np.matrix(mu, dtype=dataType))

        if sigma is None:
            cov = np.matrix(np.identity(k, dtype=dataType))
        elif type(sigma) is not np.matrix:
            cov = np.matrix(sigma, dtype=dataType)
            for i in cov.shape:  # Ensure supplied cov is square
                assert i == k
        else:
            cov = sigma

        self.dim = k
        self.mean = mu
        self.cov = cov
        self.det = np.linalg.det(cov)
        self.inv = np.linalg.inv(cov)
        self.dtype = dataType

    def evaluate(self, x):
        '''
        Evaluate the Multivariate Gaussian PDF at vector x
        return np.double
        enforces len(x) == k
        '''

        phi = np.subtract(x, self.mean)
        return np.double(np.double(1) / (np.double(2) * np.pi ** (self.dim / 2) *
                np.sqrt(self.det)) * np.exp(np.double(-0.5) *
                phi.T * self.inv * phi))

class Exponential(object):
    '''Univariate Exponential Distribution on a scalar x'''
    def __init__(self, l, **kwargs):
        self.l = l

    def evaluate(self, x):
        if x < 0:
            return 0
        else:
            return self.l * np.exp(-self.l * x)

class Uniform(object):
    '''Multivariate Uniform Distribution'''
    def __init__(self, bounds):
        '''Bounds is an array-like object containing dimensionally ranked
        min/max pairs.
        For example, a unit square:
            Uniform(np.array([[0, 1], [0, 1]]))
        '''
        self.bounds = bounds
        self.value = 1 / np.product(np.max(bounds, axis=0) -
                                np.min(bounds, axis=0), dtype=np.double)

    def in_bounds(self, x):
        '''Slow for now, will implement in boolean array eventually'''
        for i, v in enumerate(x):
            if self.bounds[i*2] <= v <= self.bounds[i*2 + 1]:
                continue
            else:
                return False
        return True

    def evaluate(self, x):
        if self.in_bounds(x):
            return self.value
        else:
            return 0.
