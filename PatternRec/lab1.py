#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import classifiers as CLSF
import sys
fig_count = 1
RES = 0.1

# Training Classes
A = CLSF.PatternClass('A', 200, (5, 10), np.matrix(((8, 0), (0, 4))), 'r')
B = CLSF.PatternClass('B', 200, (10, 15), np.matrix(((8, 0), (0, 4))), 'b')
C = CLSF.PatternClass('C', 100, (5, 10), np.matrix(((8, 4), (4, 40))), 'm')
D = CLSF.PatternClass('D', 200, (15, 10), np.matrix(((8, 0), (0, 8))), 'y')
E = CLSF.PatternClass('E', 150, (10, 5), np.matrix(((10, -5), (-5, 20))), 'g')

# Region Plotting Boundaries
CaseBounds = {1: ((-5, 20), (0, 25)), 2: ((-5, 20), (-10, 25))}
g1, g2 = (A, B), (C, D, E)


def main():
    # Show the clusters #######################################################
    showClusters(g1)  # Group 1
    showClusters(g2)  # Group 2

    # Show MED, MAP, MICD/GEM #################################################
    # Group 1
    med1 = CLSF.MED_Classifier(g1)
    map1 = CLSF.MAP_Classifier(g1)
    gem1 = CLSF.GEM_Classifier(g1)
#    showBoundaries((med1, map1, gem1), CaseBounds[1])

    # Group 2
    med2 = CLSF.MED_Classifier(g2)
    map2 = CLSF.MAP_Classifier(g2)
    gem2 = CLSF.GEM_Classifier(g2)
#    showBoundaries((med2, map2, gem2), CaseBounds[2])

    # Show KNN for k in (1, 5) ################################################
    # Group 1
    nn1 = CLSF.KNN_Classifier(g1, 1)
    knn1 = CLSF.KNN_Classifier(g1, 5)
#    showNNBoundaries((nn1, knn1), CaseBounds[1])

    # Group 2
    nn2 = CLSF.KNN_Classifier(g2, 1)
    knn2 = CLSF.KNN_Classifier(g2, 5)
    showNNBoundaries((nn2, knn2), CaseBounds[2])

    # Error Analysis ##########################################################
    print "Confusion Matrices:"
    print "{A, B} MED:\n" + str(med1.confuse())
    print "{A, B} MAP:\n" + str(map1.confuse())
    print "{A, B} GEM:\n" + str(gem1.confuse())
    print "{C, D, E} MED:\n" + str(med2.confuse())
    print "{C, D, E} MAP:\n" + str(map2.confuse())
    print "{C, D, E} GEM:\n" + str(gem2.confuse())

    print "{A, B} NN:\n" + str(nn1.confuse())
    print "{A, B} 5NN:\n" + str(knn1.confuse())
    print "{C, D, E} NN:\n" + str(nn2.confuse())
    print "{C, D, E} 5NN:\n" + str(knn2.confuse())


def showClusters(classes, plot_ellipse=True):
    '''Plots cluster groups with ellipses'''
    global fig_count
    fig = plt.figure(fig_count)
    ax = fig.add_subplot(111)
    for c in classes:
        c.display(ax, ellipse=plot_ellipse)
    ax.axis('equal')
    fig.show()
    fig_count += 1


def showBoundaries(classifiers, bounds):
    global fig_count
    fig = plt.figure(fig_count)
    ax = fig.add_subplot(111)
    for c in classifiers:
        for z in c.classes:
            z.display(ax)
        c.showRegions(bounds, ax, RES, mode='region')
    ax.axis('equal')
    fig.show()
    fig_count += 1


def showNNBoundaries(classifiers, bounds):
    '''Doesn't show ellipse for classes'''
    global fig_count
    fig = plt.figure(fig_count)
    ax = fig.add_subplot(111)
    for c in classifiers:
        for z in c.classes:
            z.display(ax, ellipse=False)
        c.showRegions(bounds, ax, RES)
    ax.axis('equal')
    fig.show()
    fig_count += 1


if __name__ == '__main__':
    sys.exit(main())
