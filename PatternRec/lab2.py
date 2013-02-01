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

    # Plot 1a: A (red) + B (blue)
    ax1 = fig.add_subplot(211)
    for cls, color in [(A, 'r'), (B, 'b')]:
        ax1.scatter(cls.samples[:, 0], cls.samples[:, 1], c=color, marker='o')
        auxPlot.plotEllipse(ax1, cls.mean, cls.cov, color)
    ax1.axis('equal')

    # Plot 1b: C (magenta), D (cyan), E (green)
    ax2 = fig.add_subplot(212)
    for cls, color in [(C, 'm'), (D, 'c'), (E, 'g')]:
        ax2.scatter(cls.samples[:, 0], cls.samples[:,1], c=color, marker='o')
        auxPlot.plotEllipse(ax2, cls.mean, cls.cov, color)
    ax2.axis('equal')

    fig.show()  # TODO: add legends




class  PatternClass():
    '''Container for training data for pattern recognition tasks
    currently hardcoded for normal distributions'''
    def __init__(self, n, mean, cov):
        self.samples = np.random.multivariate_normal(mean, cov, n)
        self.mean = np.transpose(np.matrix(mean))
        self.cov = cov



if __name__ == '__main__':
    main()
