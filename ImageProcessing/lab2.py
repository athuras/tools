import numpy as np
import skimage
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from scipy.signal import convolve2d, medfilt2d
from scipy.ndimage.filters import gaussian_filter
import matplotlib.image as mpimg
from lab1 import psnr

# Various Constants
root_dir = '/Users/ath/Desktop/syde575_lab2/'
sf_params = {'bbox_inches': 0, 'pad_inches': 0}
imshow_params = {'cmap':'gray', 'vmin':0, 'vmax':1.}

# Various
def get_data():
    path = '/Library/Python/2.7/site-packages/skimage/data'
    lena = rgb2gray(mpimg.imread(path + '/lena.png'))
    cam = mpimg.imread(path + '/camera.png')
    return lena, cam

def normalize(x):
    return x / x.max()

def array_with_hist(x):
    f, axs = plt.subplots(1, 2, figsize=(8, 5))
    axs[0].imshow(x, cmap='gray', vmin=0.0, vmax=1.)
    axs[1].hist(x.ravel(), 256, range=(0., 1.), fc='k', ec='k', normed=True)
    return f

def awh2(x):
    f, axs = plt.subplots(1, 2, figsize=(8, 5))
    axs[0].imshow(x, cmap='gray', vmin=0.0, vmax=1.)
    axs[1].hist(x.ravel(), 256, range=(0., 1.0), fc='k', ec='k', normed=True)
    return f

def gauss_kernel(size, size_y=None):
    '''Normalized 2D gauss kernel array
    Implicitely delivers square array of side= 2 * size + 1'''
    size = int(size)
    size_y = size if size_y is None else int(size_y)
    x, y = np.mgrid[-size:size + 1, -size_y:size_y + 1]
    g = np.exp(-(x**2/float(size) + y**2/float(size_y)))
    return g / g.sum()

def salt_and_pepper(f, density=0.05):
    x = f.copy()
    n1 = int(x.size * density)
    n2 = int(x.size * density / 2)
    n1 -= n2
    black = np.random.randint(0, x.size, size=n1)
    white = np.random.randint(0, x.size, size=n2)
    x.flat[black] = 0.0
    x.flat[white] = 1.0
    return x

def speckle_noise(f, std=0.2):
    '''Multiplicative uniform noise'''
    x = f.copy()
    l = 1. / np.sqrt(4. / 12) * std
    x *= np.random.uniform(-l, l, size=x.shape) + 1.
    return x

def three_by_three_unsharp(f, r):
    '''Show range of weighting parameters in 3x3 grid'''
    imgboost, axes = plt.subplots(3, 3, figsize=(12,12), sharex=True, sharey=True)
    factors = np.array([0., 0.4, 0.8,
                        1., 1.5, 3.,
                        5., 10., 100.])
    for i, e in enumerate(axes.flat):
        e.imshow(f - factors[i] * r, cmap='gray', vmin=0., vmax=1.)
        e.set_title('k = %f' % factors[i])
    return imgboost

def main():
    f = np.hstack((np.ones((200, 100))*0.3, np.ones((200, 100))*0.7))
    lena, camera = get_data()
    gauss_noised = f + np.random.normal(size=f.shape, scale=0.1)
    sp_noised = salt_and_pepper(f, density=0.05)
    specked = speckle_noise(f)

    f1 = array_with_hist(f)
    f2 = array_with_hist(gauss_noised)
    f3 = array_with_hist(sp_noised)
    f4 = array_with_hist(specked)

    noisy_lena = lena + np.random.normal(scale=0.002**.5, size=lena.shape)

    f5 = array_with_hist(noisy_lena)

    # Various Kernels
    k = np.ones((3,3), dtype=np.double) * 1. / 9.
    k7 = np.ones((7,7), dtype=np.double)*1./49.
    g = gauss_kernel(3)

    denoised = convolve2d(noisy_lena, k, mode='same')
    denoised2 = convolve2d(noisy_lena, k7, mode='same')

    f5 = array_with_hist(lena)
    f6 = array_with_hist(noisy_lena)
    f7 = array_with_hist(denoised)
    f8 = array_with_hist(denoised2)
    zz = convolve2d(noisy_lena, g, mode='same')
    f15 = array_with_hist(zz)  # Apologies for order change.

    # Salt and Pepper
    salted = salt_and_pepper(lena)
    avg_salt = convolve2d(salted, k7, mode='same')
    norm_salt = convolve2d(salted, g, mode='same')
    med_lena = medfilt2d(salted, 7)

    f9 = array_with_hist(salted)
    f10 = array_with_hist(avg_salt)
    f11 = array_with_hist(norm_salt)
    f12 = array_with_hist(med_lena)

    # Sharpening:
    # 2x2 Camera: Original, Blurred, Residual, Boosted
    g_camera = convolve2d(camera, g, mode='same')
    sharper, axes = plt.subplots(2, 2, sharey=True, figsize=(8,8))
    ims = (camera, g_camera, r, camera + r)
    labels = ('original', 'gaussian blurred', 'residual', 'boosted')
    for i, e in enumerate(axes.flat):
        e.imshow(ims[i], cmap='gray', vmin=0., vmax=1.)
        e.set_title(labels[i])
    axes.flat[2].imshow(r, cmap='gray_r')

    f13 = sharper
    f14 = three_by_three_unsharp(camera, r)

    print 'fin'
