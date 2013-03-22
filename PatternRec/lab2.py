#!/usr/bin/python
import numpy as np
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
import estimators as est
import pdfs
from classifiers import ML_Classifier
import seq

RES = 5
fig_count = 0
debug = [4]

def main():
    global fig_count
    m1, m2, m3 = map(sio.loadmat, ('mat/lab2_1', 'mat/lab2_2', 'mat/lab2_3'))
    a, b = map(np.ravel, (m1['a'], m1['b']))
    al, bl, cl = m2['al'], m2['bl'], m2['cl']
    a2, b2 = m3['a'], m3['b']

    REF_A = pdfs.Multivariate_Normal(1, np.matrix([5]), np.matrix([1]))
    REF_B = pdfs.Exponential(1)

    # Part 1
    if 0 in debug:
        B = (-2, 12)
        gaussA = est.multivariate_normal(a, 1)
        expA = est.univariate_exponential(a)
        uniA = est.multivariate_uniform(a)
        kdeA1 = est.KDE(a, 1, np.matrix([0.1]))
        kdeA2 = est.KDE(a, 1, np.matrix([0.4]))

        gaussB = est.multivariate_normal(b, 1)
        expB = est.univariate_exponential(b)
        uniB = est.multivariate_uniform(b)
        kdeB1 = est.KDE(b, 1, np.matrix([0.1]))
        kdeB2 = est.KDE(b, 1, np.matrix([0.4]))

        [plot_multiple_1d((z.evaluate, REF_A.evaluate), j, B)
          for z, j in ((gaussA, 'ML Normal'),
                      (expA, 'ML Exponential'),
                      (uniA, 'ML Uniform'))]
        [plot_multiple_1d((z.evaluate, REF_A.evaluate), j, B)
              for z, j in ((kdeA1, 'KDE, S=0.1'),
                           (kdeA2, 'KDE, S=0.4'))]
        B = (-5, 10)
        [plot_multiple_1d((z.evaluate, REF_B.evaluate), j, B)
          for z, j in ((gaussB, 'ML Normal'),
                      (expB, 'ML Exponential'),
                      (uniB, 'ML Uniform'))]
        [plot_multiple_1d((z.evaluate, REF_B.evaluate), j, B)
              for z, j in ((kdeB1, 'KDE, S=0.1'),
                           (kdeB2, 'KDE, S=0.4'))]
        plt.show()


    # 2D stuff
    # Parametrics:
    B2 = ((0, 450), (0, 450))
    if 1 in debug:
        classes = []
        for c in (al, bl, cl):
            mu, S = np.mean(c, axis=0), np.cov(c.T)
            classes.append(pdfs.Multivariate_Normal(2, mu, S))
        ML = ML_Classifier(classes)

        fig = plt.figure(fig_count)
        fig_count += 1
        ax = fig.add_subplot(111)
        for data, color in ((al, 'r'), (bl, 'b'), (cl, 'g')):
            ax.scatter(data[:,0], data[:, 1], c=color, marker='o')
        ax.legend(('Class A', 'Class B', 'Class C'), scatterpoints=1)
        iterRegion(B2, ax, ML.classify, RES)
        plt.show()

    # Non-Parametric
    if 2 in debug:
        fig_count
        fig = plt.figure(fig_count)
        fig_count += 1
        ax = fig.add_subplot(111)
        for data, color in ((al, 'r'), (bl, 'b'), (cl, 'g')):
            ax.scatter(data[:,0], data[:,1], c=color, marker='o')

        kernels = [est.KDE(c, 2, np.cov(c.T)) for c in (al, bl, cl)]
        ML = ML_Classifier(kernels)
        iterRegion(B2, ax, ML.classify, RES)
        ax.legend(('Class A', 'Class B', 'Class C'), scatterpoints=1)
        plt.show()

    # Sequential Descriminants
    B3 = ((100, 500), (0, 450))
    data = np.concatenate((a2, b2))
    labels = np.concatenate((np.zeros(len(a2), dtype=np.int),
                                np.ones(len(b2), dtype=np.int)))
    if 3 in debug:
        clsfs = [seq.seq_Linear(data, labels) for i in xrange(3)]
        for c in clsfs:
            c.iter_train()
        plot_seqs((a2, b2), B3, clsfs, RES)
        plt.show()

    if 4 in debug:
        # Error stuff.
        errors = []
        for i in xrange(5):
            # We can reuse object, and retrain
            trials = []
            S = seq.seq_Linear(data, labels)
            for p in xrange(20):
                S.iter_train(i + 1)
                trials.append(S.fit_performance())
            trials = np.array(map(lambda x: float(x[0]) / x[1], trials))
            errors.append((np.mean(trials),
                           np.std(trials),
                           np.min(trials),
                           np.max(trials)))

        bar_labels = map(str, range(1, 6))
        width = 0.2
        ind = range(len(errors))
        fig = plt.figure(fig_count)
        fig_count += 1
        ax = fig.add_subplot(111)
        c = ['b', 'g', 'r', 'c', 'm']
        bars = [ax.bar(map(lambda z: z + k*width, ind), [e[k] for e in errors], width, color=c[k])
                for k in xrange(len(errors[0]))]
        ax.set_xticks(map(lambda z: z + 2*width, ind))
        ax.set_xticklabels(bar_labels)
        ax.legend([x[0] for x in bars], ('Mean Error', 'Std Error', 'Min Error', 'Max Error'))
        plt.show()


def plot_seqs(data, b, seqs, h, **kwargs):
    global fig_count
    colors = ['r', 'b', 'g', 'o']
    for S in seqs:
        fig = plt.figure(fig_count)
        fig_count += 1
        ax = fig.add_subplot(111)
        for i, d in enumerate(data):
            ax.scatter(d[:,0], d[:,1], c=colors[i])
        S.iterRegion(b, ax, h)
        ax.legend(('Class A', 'Class B', 'Class C'), scatterpoints=1)
    return None

def iterRegion(b, splot, fn, h, **kwargs):
    '''Execute fn across region, spray results to a splot obkect,
    fn is passed a column vector'''
    colorMap = kwargs.get('cmap', matplotlib.colors.ListedColormap(['r', 'b', 'g']))
    xx, yy = np.meshgrid(np.arange(b[0][0], b[0][1], h),
                         np.arange(b[1][0], b[1][1], h))
    xn, yn = xx.shape
    Z = np.zeros(xx.shape)
    for ix in xrange(xn):
        for iy in xrange(yn):
            Z[ix, iy] = fn(np.c_[xx[ix, iy], yy[ix,iy]].T)
    splot.contour(xx, yy, Z, cmap=colorMap)
    return None

def plot_multiple_1d(fns, label,  bounds):
    global fig_count
    fig = plt.figure(fig_count)
    fig_count += 1
    ax = fig.add_subplot(111)
    x = np.arange(bounds[0], bounds[1], 0.05)
    for f in fns:
        y = np.array([f(np.array([j])) for j in x])
        ax.plot(x, y)
        ax.legend((label, "True PDF"))
    ax.axis('auto')
    return None

if __name__ == '__main__':
    main()
