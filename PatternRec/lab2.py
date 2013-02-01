#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import plot as auxPlot


def main():
    # Set up the class data
    A = PatternClass(200, (5, 10), np.matrix(((8, 0), (0, 4))))
    B = PatternClass(200, (10, 15), np.matrix(((8, 0), (0, 4))))
    C = PatternClass(100, (5, 10), np.matrix(((8, 4), (4, 40))))
    D = PatternClass(200, (15, 10), np.matrix(((8, 0), (0, 8))))
    E = PatternClass(150, (10, 5), np.matrix(((10, -5), (-5, 20))))

    fig = plt.figure()

    # Plot 1: A (red) + B (blue)
    ax1 = fig.add_subplot(111)
    ax1.scatter(A.samples[:,0],
                A.samples[:,1],
                c='r', marker='o')
    ax1.scatter(B.samples[:,0],
                B.samples[:,1],
                c='b', marker='o')
    auxPlot.plotEllipse(ax1, A.mean, A.cov, 'r')
    auxPlot.plotEllipse(ax1, B.mean, B.cov, 'b')
    ax1.axis('equal')

    fig.show()



class  PatternClass():
    '''Container for training data for pattern recognition tasks
    currently hardcoded for normal distributions'''
    def __init__(self, n, mean, cov):
        self.samples = np.random.multivariate_normal(mean, cov, n)
        self.mean = np.transpose(np.matrix(mean))
        self.cov = cov



if __name__ == '__main__':
    main()
