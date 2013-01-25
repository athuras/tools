#!/usr/bin/env python
'''Reservoir sample (by line) a stream'''
import sys
from numpy import random


def main(args=None):
    n, c = 0, 0
    if args is None:
        n = 10
    else:
        n = int(args[0])

    R = []

    for l in sys.stdin:
        # Fill the reservoir
        if c < n:
            R.append(l)
        else:
            r = random.randint(0, c)
            if r < n:
                R[r] = l
        c += 1

    for record in R:
        sys.stdout.write(record)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
