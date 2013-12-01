import numpy as np
import random
import heapq as H

class pTuple(tuple):
    '''
    Subclass of tuple where '>' and '<' only look at the first element,
    useful in a heapq list.

    This was necessary for the implementation in HarmonySearch, wherein the
    'data' for each heap element was implicitely being compared (which becomes
    annoying for np.array).
    '''
    def __lt__(self, other):
        return self[0] < other[0]
    def __gt__(self, other):
        return self[0] > other[0]

def in_bounds(bounds, x):
    for i, e in enumerate(x):
        if not bounds[i][0] <= e <= bounds[i][1]:
            return False
    return True

class HarmonySearch(object):
    '''
    The Harmony Search Hill-Climbing algorithm,
    without fret width discretization
    '''
    def __init__(self, f, bounds, constraints=[], hms=10, **kwargs):
        '''
        Set the various parameters for the Harmony Search Metaheuristic
        Optimization Method.
        * f: the objective function to be MAXIMISED. input should be a 1xN array
        * bounds: the region to be explored i.e. ((-10, 10), (10, 20)).
        * constraints: an iterable of functions that have the following
            signature-- (scalar, 1-d array) -> Bool
        * hms: Harmony Memory Size, the size of the lookup heap

        kwargs:
        ** max_iter: The number of iterations to perform before exit
        ** hmcr: Harmony Memory Criteria Rate, the probability that a solution
            will be drawn from the existing bank
        ** par: Pitch Adjustment Rate, given hmcr, the probability that the
            memoized solution will be adjusted by the fret width (fw)
        ** fw: Fret Width, broadcastable quantity to add/subtract from a
            memoized solution to find a new solution
        ** pitch_sequence: A generator that yields an infinite series of
            booleans. Controls whether pitch is adjusted up or down,
            Defaults to a flip-flop (True, False, True, False ...)

        Example Usage:
        def f(x):
            return np.cos(x.sum(axis=1))

        hs = HarmonySearch(f, ((-np.pi, pi), (-np.pi, pi)), hms=10)
        hs.execute()
        >> array([ 0.72568394, -0.76400488])
        '''
        self.hms = hms  # Harmony Memory Size
        self.constraints = constraints
        self.bounds = bounds
        self.constraints.append(lambda x: in_bounds(self.bounds, x[1]))
        self.hm = []
        self.f = f

        # Additional Parameters
        self.max_iter = kwargs.get('max_iter', 100)
        self.fw = kwargs.get('fw', 0.1)
        self.hmcr = kwargs.get('hmcr', 0.6)
        self.par = kwargs.get('par', 0.05)
        self.pitch_sequence = kwargs.get('pitch_sequence', HarmonySearch.alternator())

    def random_selection(self):
        return [random.uniform(i[0], i[1]) for i in self.bounds]

    def random_selection_as_array(self):
        return  np.array(self.random_selection())

    @staticmethod
    def alternator(i=True):
        '''Yeeeeeeeehaw!'''
        while True:
            yield i
            i = not i

    def initial_improvise(self):
        return [self.random_selection_as_array() for i in xrange(self.hms)]

    def next_note(self):
        '''
        Decide which 'note' to play next probabilistically,
        returns (f(n), n) pTuple.
        See harmony.pTuple for details.
        '''
        if len(self.hm) < self.hms:
            s = self.random_selection_as_array()
            return pTuple((self.f(s[None, :]), s))

        d = random.uniform(0., 1.)
        if d > self.hmcr:
            s = self.random_selection_as_array()
            return pTuple((self.f(s[None, :]), s))

        z = random.randint(0, self.hms - 1)
        if d < self.hmcr * self.par:
            n = None
            if next(self.pitch_sequence):
                n = self.hm[z][1] + self.fw
            else:
                n = self.hm[z][1] - self.fw
            return pTuple((self.f(n[None, :]), n))
        else:
            return self.hm[z]

    def execute(self):
        for i in xrange(self.max_iter):
            n = self.next_note()
            if not all(z(n) for z in self.constraints):
                continue

            if len(self.hm) < self.hms:
                H.heappush(self.hm, n)
            elif n[0] >= self.hm[0][0]:
                _ = H.heappushpop(self.hm, n)

        return max(self.hm, key=lambda x: x[0])[1]
