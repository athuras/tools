#!/usr/bin/env python
import numpy as np


def main():
    pass

class fast_MED(object):
    '''almost array-native MED classifier'''
    def __init__(self, prototypes, **kwargs):
        '''Prototypes is a list of row-vectors, index implies class id'''
        self.prototypes = prototypes

    def classify(self, points):
        '''points is a matrix of row-vectors, returns array of classification indices'''
        # First get the distances
        distances = [np.sum(np.multiply(points - x, points - x), axis=1)
                        for x in self.prototypes]
        stack = np.dstack(distances)  # Lets get serious
        return np.ravel(np.argmin(stack, axis=2))

    def confuse(self, points, labels):
        '''Returns a nxn confusion matrix with respect to labels.
        example: med.confuse(np.identity(2), np.array([1, 0]))
        => np.array([[0,1],[1,]])  # Perfect failure!

        [results, confusion] = confuse(stuff)
        '''
        if len(points) != labels.size:
            raise AssertionError
        k = len(self.prototypes)
        results = self.classify(points)
        return results, np.bincount(k * labels + results, minlength=k*k).reshape(k, k)

class seq_Linear(object):
    '''Random sequential linear BINARY classifier, I dearly want to generalise
    this, but that time will come ...'''
    def __init__(self, data, labels, **kwargs):
        '''Data is a list of row-vector arrays,
        labels is a list of 1d-int_array identities,
        the order of data and labels should be consistent.
        kwargs: iter_limit, max iterations before break'''

        iter_limit = kwargs.get('iter_limit', 1000)
        self.ITER_LIMIT = iter_limit
        self.data = data
        self.labels = labels
        self.classifiers = []
        self.attempts = 0

    def iter_train(self, max_cls=100):
        '''Thrashes until it linearly separates points, this will hit the
        iter_limit (and break) if training points are not linearly separable'''
        # Wastes a little memory here, but whose counting?

        def remove_idx(state, idx):
            return [np.delete(z, idx, axis=0) for z in state]

        def get_prototype(state, prots = (0, 1)):
            '''Randomly select a list of vector prototypes from state'''
            # This is ugly, I'm sorry
            idx = [np.random.choice(np.where(state[1] == i)[0], 1)
                   for i in prots]
            return [np.take(state[0], i, axis=0) for i in idx]

        unres = [np.copy(self.data), np.copy(self.labels)]
        sequence = []
        current_iteration = None
        for current_iteration in xrange(self.ITER_LIMIT):
            if unres[0].size == 0:
                break
            G = fast_MED(get_prototype(unres, (0, 1)))
            results, C = G.confuse(unres[0], unres[1])
            if C[0,1] != 0 and C[1,0] != 0:
                continue

            R = []
            if C[0,1] == 0:
                R.append(np.where(results == 1)[0])
            if C[1,0] == 0:
                R.append(np.where(results == 0)[0])
            R = np.concatenate(R)
            unres = remove_idx(unres, R)
            sequence.append((G, C))
            if len(sequence) >= max_cls:
                break

        self.classifiers = sequence
        self.attempts = current_iteration


    def classify(self, x):
        '''Run down the sequence, could return horrible horrible things if
        not linearly seperable. This is sadly NOT array-native, also only takes a
        SINGLE vector'''
        rt = None
        for clsf, C in self.classifiers:
            rt = clsf.classify(x)
            if (rt == 0 and C[1,0] == 0) or (rt == 1 and C[0,1] == 0):
                return rt
            else:
                continue
        return rt  # Degenerates with non-perfect assesment

    def iterRegion(self, bounds, splot, h=1, **kwargs):
        xx, yy = np.meshgrid(np.arange(bounds[0][0], bounds[0][1], h),
                             np.arange(bounds[1][0], bounds[1][1], h))
        xn, yn = xx.shape
        Z = np.zeros(xx.shape)
        for ix in xrange(xn):
            for iy in xrange(yn):
                Z[ix, iy] = self.classify(np.c_[xx[ix, iy], yy[ix, iy]])
        splot.contour(xx, yy, Z)
        return splot

    def fit_performance(self):
        '''Returns (incorrect_count, total_count)'''
        assert len(self.classifiers) > 0
        total = np.sum(self.classifiers[0][1])
        incorrect = (np.sum(self.classifiers[-1][1]) -
                     np.trace(self.classifiers[-1][1]))
        return incorrect, total

    def confuse(self, data, labels):
        pass


if __name__ == '__main__':
    main()
