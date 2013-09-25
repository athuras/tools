#!/usr/bin/env python
from scipy.ndimage import zoom
from skimage import exposure
from skimage.color import rgb2gray
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.signal import convolve

def mse(f, g):
    '''Mean Squared Error between images f, g
    * reindexed to the smaller image because fuck you MATLAB'''
    return np.square(f - g).sum() / f.size

def psnr(f, g):
    '''Peak Signal to Noise Ratio'''
    m = mse(f, g)
    return 10 * np.log10(f.max() / m)

def run_quality_stats(f, g):
    return {'mse': mse(f, g), 'psnr': psnr(f, g)}

def get_data():
    path = '/Library/Python/2.7/site-packages/skimage/data'
    lena = rgb2gray(mpimg.imread(path + '/lena.png'))
    camera = mpimg.imread(path + '/camera.png')
    tire = mpimg.imread('/Users/ath/Downloads/tire.png')
    return lena, camera, tire

def downsample(img, z=0.25, o=0):
    '''Using bi-polynomial interpolation of order-'o' zoom the image'''
    return zoom(img, z, order=o)


class Lab1(object):
    def __init__(self, images):
        self.images = images
        figs = []
        figs.append(Lab1.fig1_original_images(images[:2]))

        f2, downsampled = Lab1.fig2_downsampled_images(images[:2])
        figs.append(f2)

        f3, degraded = Lab1.fig3_upsampled_images(downsampled)
        figs.append(f3)

        stats, deltas = Lab1.run_stats(images[:2], degraded)
        figs.append(Lab1.fig4_show_deltas(deltas))
        self.deltas = deltas
        self.stats = stats

        # Part 4: Discrete Convolution
        filters = [np.ones((1, 6), dtype=np.float64) / 6,
                   np.ones((6, 1), dtype=np.float64) / 6,
                   np.array([[-1, 1]], dtype=np.float64)]
        figs.append(Lab1.convolutions(self.images[0], filters))

        # Part 5: Points Operations for Image Enhancement
        # Histograms
        figs.extend(Lab1.histograms_of_stuff(images[-1]))
        self.figs = figs

    @classmethod
    def histograms_of_stuff(cls, ref):
        # First we show the image, and its histogram
        neg = 1. - ref
        figs = []
        f1, axs1 = plt.subplots(2, 2, figsize=(8, 8))
        for i, im in enumerate(((ref, 'tire'), (neg, 'negative'))):
            axs1[i,0].imshow(im[0], cmap='gray')
            axs1[i,1].hist(im[0].ravel(), 256, range=(0., 1.), fc='k', ec='k')
            axs1[i,0].set_title(im[1])
            axs1[i,1].set_title(im[1] + ': histogram')

        figs.append(f1)

        # Some gamma correction ...
        gammas = (0.5, 1.3)
        f2, axs2 = plt.subplots(2, 2, figsize=(8,8))
        for i, gamma in enumerate(gammas):
            x = ref ** gamma
            axs2[i,0].imshow(x, cmap='gray')
            axs2[i,1].hist(x.ravel(), 256, range=(0.,1.), fc='k', ec='k')
            for z in axs2[i]:
                z.set_title('gamma = ' + str(gamma))
        figs.append(f2)

        # Histogram Equalization
        f3, axs3 = plt.subplots(1, 2, figsize=(8, 4))
        eq_ref = exposure.equalize_hist(ref)
        axs3[0].imshow(eq_ref, cmap='gray')
        axs3[1].hist(eq_ref.ravel(), 256, range=(0.,1.), fc='k', ec='k')
        for x in axs3:
            x.set_title('Histogram Equalization')
        figs.append(f3)
        return figs

    @classmethod
    def convolutions(cls, ref, fs):
        convs = [convolve(ref, f) for f in fs]
        f, axs = plt.subplots(2, 2, sharey=True,
                              figsize=(8, 8))
        axs[0,0].imshow(ref, cmap='gray')
        axs[0,0].set_title('Original')
        titles = ('H1', 'H2', 'H3')

        for i, a in enumerate((axs[0,1], axs[1,0], axs[1,1])):
            a.imshow(convs[i], cmap='gray')
            a.set_title(titles[i])
            a.set_aspect('equal')
        return f

    @classmethod
    def fig1_original_images(cls, images):
        m = len(images)
        f = plt.figure(figsize=(m * 4, 4))
        for i, im in enumerate(images):
            a = f.add_subplot(1, m, i + 1)
            a.imshow(im, cmap='gray')
        return f

    @classmethod
    def fig2_downsampled_images(cls, images, r=0.25, o=0):
        m = len(images)
        f = plt.figure(figsize=(m * 4, 2))
        arrays = []
        for i, im in enumerate(images):
            a = f.add_subplot(1, m, i + 1)
            x = zoom(im, r, order=o)
            a.imshow(x, cmap='gray')
            arrays.append(x)
        return f, arrays

    @classmethod
    def fig3_upsampled_images(cls, images, r=4., orders=(0, 1, 3)):
        '''Creates an images by orders grid'''
        omap = {0: 'nearest',
                1: 'bi-linear',
                2: 'bi-quadratic',
                3: 'bi-cubic',
                4: 'bi-quartic',
                5: 'bi-quintic'}

        n, m = len(images), len(orders)
        f, axs = plt.subplots(n, m, sharex=True, sharey=True,
                              figsize=(n * 4, m * 2))
        arrays = []
        for i, im in enumerate(images):
            arrays.append([])
            for j, o in enumerate(orders):
                x = zoom(im, r, order=o)
                axs[i, j].imshow(x, cmap='gray')
                axs[i, j].set_title(omap[o])
                axs[i, j].set_aspect('equal')
                arrays[i].append(x)
        return f, arrays

    @classmethod
    def fig4_show_deltas(cls, deltas):
        omap = {0: 'nearest',
                1: 'bi-linear',
                2: 'bi-quadratic',
                3: 'bi-cubic',
                4: 'bi-quartic',
                5: 'bi_quintic'}
        n, m = len(deltas), 3
        f, axs = plt.subplots(n, m, sharex=True, sharey=True,
                              figsize=(n * 4, m * 2))
        for i, e in enumerate(deltas):
            for j, im in enumerate(e):
                axs[i, j].imshow(im, cmap='gray_r')
                axs[i, j].set_title(omap[j])
                axs[i, j].set_aspect('equal')
        return f

    @classmethod
    def run_stats(cls, originals, images):
        stats, deltas = [], []
        for i, im in enumerate(originals):
            stats.append([])
            deltas.append([])
            for j, g in enumerate(images[i]):
                stats[i].append(run_quality_stats(im, g))
                deltas[i].append(np.abs(g - im))
        return stats, deltas

def run_lab_one():
    '''Initializes the lab object, generates the plots etc.'''
    data = get_data()
    lab = Lab1(data)
    # Enjoy!
