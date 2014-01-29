import numpy as np


def force_array(iterable):
    '''GET BACK SATAN!'''
    return [np.array(x) if type(x) is not np.ndarray else x for x in iterable]

def collapse(A, k=1):
    '''Reshape array of dimension D = len(A.shape) to D -k by
    tesselating the last dimension on the previous one.
    If k > D, this is equivalent (in effect) to np.flatten
    '''
    if len(A.shape) <= k:
        k = len(A.shape) - 1
    S = list(A.shape)
    for i in xrange(k):
        q = S.pop()
        S[-1] *= q
    return np.reshape(A, tuple(S))

def dot_expand(x, y):
    '''Explodes the dimensions of the result based on the dimensions of input'''
    if len(x.shape) == 1 and len(y.shape) == 1:
        return np.dot(np.c_[x], y[None, ...])
    else:
        return np.dot(x, y.T)

def diag_dot(x, y):
    return np.c_[x * y].sum(axis=-1)
