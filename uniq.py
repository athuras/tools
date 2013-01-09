#!/usr/bin/env python
'''
Written as a more histogram-oriented uniq tool

Usage:
    cat some_file | python uniq.py '[1,2,3]'
    key is interpreted as a bunch of integer keys in a tab-separated-file
'''
from collections import defaultdict
import sys

def main(args):
    # Establish Sets
    key = map(int, args)
    uniqs = defaultdict(lambda: set())
    for l in sys.stdin:
        f = l.rstrip('\n').split('\t')
        for z in key:
            uniqs[z].add(f[z])

    # Reallocate
    for k in uniqs.iterkeys():
        uniqs[k] = len(uniqs[k])

    # Display results
    print '\t'.join(map(str, key)) + '\tUniques'
    for k, v in uniqs.iteritems():
        print '\t'.join(map(str, [k, v]))


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
