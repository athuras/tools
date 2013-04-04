#!/usr/bin/env python
# Note Figure Order is somewhat shuffled.

import fast_classifiers as fst
import numpy as np
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt


h = 0.001

def main():
    # load and prepare the data
    # Image Classification and Segmentation
    k = sio.loadmat('mat/lab3/feat.mat')
    data = prepare_data([k['f2'].T, k['f8'].T, k['f32'].T])
    test = prepare_data([k['f2t'].T, k['f8t'].T, k['f32t'].T])
    classifiers = [fst.MICD(*x) for x in data]
    confusion = [x.confuse(*test[i]) for i, x in enumerate(classifiers)]
    for z in confusion:
        print z
        print p_error(z)

    # Segmentation with plots
    micd8 = classifiers[1]
    multf8 = k['multf8']
    s = multf8.shape
    result = micd8.classify(multf8.reshape((s[0]*s[1], s[2])))
    result = result.reshape((s[0], s[1]))
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.imshow(result, cmap='spectral')
    ax.axis('equal')
    ax.set_title('MICD Classification of Image Pixels')

    # K-means stuff
    k32 = data[2][0]
    km32 = fst.KMeans(k32, 10)
    km32.execute(500)
    fig = plt.figure(3)
    ax = fig.add_subplot(111)
    ax.scatter(km32.means[:,0], km32.means[:,1], marker='o', c='r')
    ax.scatter(k32[:,0], k32[:,1], marker='x', c='0.75')
    ax.axis('equal')
    ax.set_title('f32 Feature Space, with 10-Means Cluster Heads')

    # Now showing Voronoi Diagram
    xx, yy = np.meshgrid(np.arange(-0.02, 0.15, h),
                         np.arange(-0.03, 0.18, h))
    Q = np.dstack((xx, yy))
    Q = Q.reshape(Q.shape[0]*Q.shape[1], Q.shape[2])
    Z = km32.classify(Q).reshape(xx.shape)
    fig = plt.figure(6)
    ax = fig.add_subplot(111)
    ax.scatter(k32[:, 0], k32[:,1], marker='o', c='b')
    ax.pcolormesh(xx, yy, Z, alpha=0.2, cmap='spectral')
    ax.axis('auto')
    ax.set_title("Voronoi Diagram for 10-Means Clustering")

    # Show clustered image using multf8
    # Here we 'train' using the full 10-image set
    s = multf8.shape
    X = multf8.reshape((s[0]*s[1], s[2]))
    km10 = fst.KMeans(X, 10)
    km10.execute(500)
    fig = plt.figure(4)
    ax = fig.add_subplot(111)
    result = km10.classify(multf8.reshape((s[0]*s[1], s[2])))
    result = result.reshape((s[0], s[1]))
    ax.axis('equal')
    ax.set_title('Image Segment Clustering With 10-Means')
    ax.imshow(result, cmap='spectral')

    # Here we demonstrate the power of knowing how many clusters there are
    km5 = fst.KMeans(X, 5)
    km5.execute(500)
    fig = plt.figure(5)
    ax = fig.add_subplot(111)
    result = km5.classify(X)
    result = result.reshape((s[0], s[1]))
    ax.axis('equal')
    ax.set_title('Direct Image Clustering With 5-means')
    ax.imshow(result, cmap='spectral')

    plt.show()

def p_error(conf):
    return np.trace(conf, dtype=np.double) / np.sum(conf)

def prepare_data(data):
    master = []
    for datum in data:
        features = datum[:, 0:2]
        labels = np.array(datum[:, 2] - 1, dtype=np.int)
        master.append([features, labels])
    return master

if __name__ == '__main__':
    main()
