
# # SYDE 575: Lab 4 - Restoration

# In[72]:

get_ipython().magic(u'pylab --no-import-all inline')
import numpy as np
from numpy.fft import fft, ifft
from numpy.fft import fftshift, ifftshift
from numpy.fft import fft2, ifft2

import skimage
import scipy as sp
from scipy.signal import wiener, deconvolve, convolve2d
from scipy.ndimage.filters import generic_filter
from scipy.io.matlab import loadmat
from scipy.ndimage.filters import gaussian_filter

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from skimage.color import rgb2gray
from skimage.transform import rotate

from itertools import izip
from lab1 import psnr

from bottleneck import argpartsort


# Out[72]:

#     Populating the interactive namespace from numpy and matplotlib
#

# In[17]:

def get_data(skimage_resource_path='/usr/local/lib/python2.7/site-packages/skimage/data',
             degraded_path='/Users/ath/Desktop/degraded.png'):
    '''Returns the two images we'll use in the lab,
    Make sure to change the path to where the source images can be found'''
    cam = mpimg.imread(skimage_resource_path + '/camera.png')
    cam = skimage.transform.resize(cam, (256, 256))
    degraded = mpimg.imread(degraded_path)
    return cam, degraded

def ideal_mask_lp_filter(shape, cutoff):
    '''Returns a disk-filter, to be used as the user sees fit...'''
    x = np.linspace(-shape[0] / 2, shape[0] / 2., shape[0])
    y = np.linspace(-shape[1] / 2, shape[1] / 2., shape[1])
    X, Y = np.meshgrid(x, y)
    Z = np.hypot(X, Y)
    Z.reshape(shape)
    Z[Z < cutoff] = 1.
    Z[Z >= cutoff] = 0.
    return Z

def fspecial_disk(radius):
    edge = 2*radius + 1
    return ideal_mask_lp_filter((edge, edge), radius)

def fspecial_MATLAB():
    '''Returns a 9x9 disk'''
    x = loadmat('/Users/ath/Downloads/h.mat')['h']
    return x

def apply_gauss_noise(x, u, s):
    g = np.random.normal(u, s, size=x.shape)
    return x + g

def multi_plot(nrows, ncols, data, titles, sharex=False,
               sharey=False, master_cmap='gray', fs=(2,2)):
    f, axes = plt.subplots(nrows=nrows, ncols=ncols,
                           sharex=sharex, sharey=sharey, figsize=fs)
    for (ax, d, title) in izip(axes.flat, data, titles):
        ax.imshow(d, cmap='gray', vmin=0.)
        ax.set_title(title)
    return f

def deconvolve2d(x, h):
    xf, hf = fft2(x), fft2(h, x.shape)
    return np.abs(ifft2(xf / hf))



# In[18]:

camera, degraded = get_data()


# ## Part 1 – Image Restoration in the Frequency Domain

# In[19]:

# Disk Function
d = fspecial_MATLAB()
df, camf = fft2(d, camera.shape), fft2(camera)
g_blur = np.abs(ifft2(df * camf))
g_unblur = np.abs(ifft2(fft2(g_blur) / df))


# In[35]:

titles = ('Disk Blur', 'Inverse Blur: PSNR = %.2f' % psnr(camera, g_unblur))
data = (g_blur, g_unblur)
f1 = multi_plot(1, 2, data, titles, sharey=True, fs=(8, 4))


# Out[35]:

# image file:

# Note the insanely high PSNR, this should be expected, as the MSE value between the inverse filtered image and the original is negligible.

# In[21]:

# Adding Gaussian Noise
noised = apply_gauss_noise(g_blur, 0., np.sqrt(0.002))
nf = fft2(noised)
n_unblur = np.abs(ifft2(nf / df))


# In[22]:

titles = ('Inverse Blur: PSNR = %.3f' % psnr(camera, g_unblur),
          'Inverse Blur (With Noise): PSNR = %.3f' % psnr(camera, n_unblur))
data = (g_unblur, n_unblur)
f2 = multi_plot(1, 2, data, titles, sharey=True, fs=(8, 4))


# Out[22]:

# image file:

# Note the artifacts induced by the noise deblurring. Good job, lets continue.

# ### Wiener Filtering

# scipy doesn't really have a direct equivalent to the MATLAB *deconvwnr(x, h, n)* function, which takes the array *x*, the convolution kernel *h* AND the noise power *n*.
#
# scipy has the *wiener(x, n)*, which only removes noise (doesn't take information about the PSF/Kernel. This however can be (perhaps) combined with *signal.deconvolve(x, h)*. However, I believe that *wiener(deconvolve(x, h)[0], n)* is quivalent to *deconvwnr(x, h, n)* and at this point in time I'm not willing to spend the time figuring out how *deconvwnr* is implemented.

# In[23]:

def deconvwnr(x, h, n=None):
    '''???????'''
    return wiener(deconvolve2d(x, h), n)


# In[24]:

w_unblur = wiener(g_blur, 5)
w_unnoise = wiener(noised, 5)
titles = ('Inverse Blur: \nPSNR = %.3f' % psnr(camera, g_unblur),
          'Wiener Filter: \nPSNR = %.3f' % psnr(camera, w_unblur),
          'Wiener Filter (with Noise): \nPSNR = %.3f' % psnr(camera, w_unnoise),
          'w2')
data = (g_unblur, w_unblur, w_unnoise)
f3 = multi_plot(1, 3, data, titles, sharey=True, fs=(12, 4))


# Out[24]:

# image file:

# Well that was fun. Moving on...

# ## Part 2 — Adaptive Filtering

# In[25]:

# We should identify a flat region in the image to sample.
sample = degraded[20:60, 160:200]
f4, axes = plt.subplots(1, 2, sharey=False)
axes[0].imshow(degraded, cmap='gray')
axes[0].add_patch(plt.Rectangle((180, 40), 40, 40))
axes[0].set_title('Degraded\n (Sample shown in Blue)')
axes[1].imshow(sample, cmap='gray')
axes[1].set_title('Sampled Area\nVariance = %.3f' % sample.var())


# Out[25]:

#     <matplotlib.text.Text at 0x10c56bf50>

# image file:

# In[26]:

def lee_filter(x, variance_threshold, window=5):
    '''THE POWER OF ARRAY BROADCASTING COMPELS YOU'''
    u = generic_filter(x, lambda q: q.mean(), size=window)
    s2 = generic_filter(x, lambda q: q.var(), size=window)

    K = 1. - variance_threshold / s2  # A little wasteful, but fine for now...
    K[variance_threshold > s2] = 0.
    return K * x + (1. - K) * u


# In[67]:

devs = [0.333, 0.0333, 0.01]
lee_cam = lee_filter(degraded, sample.var())
titles = (['Lee Filtered Image\n $\sigma^2_n = 0.01$'] +
          ['Gaussian LPF, $\sigma=%.4f$' % i for i in devs])
gaussians = [gaussian_filter(degraded, (i, i)) for i in devs]
data = [lee_cam] + gaussians
f6 = multi_plot(2, 2, data, titles, sharey=True, sharex=True, fs=(8, 8))


# Out[67]:

# image file:

# In[53]:

plt.imshow(gaussian_filter(degraded, (3., 3.)), cmap='gray')


# Out[53]:

#     <matplotlib.image.AxesImage at 0x1073acdd0>

# image file:

# In[ ]:




# In[31]:

ws = (3, 5, 7)
vs = (0.1, 0.01, 0.001)
titles = ['(window, n_variance)\n(%i, %f)' % (x, y) for x in ws for y in vs]
data = [lee_filter(degraded, v, w) for w in ws for v in vs]


# In[32]:

f5, axes = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(12, 12))
for (a, im, t) in izip(axes.flat, data, titles):
    a.imshow(im, vmin=0., cmap='gray')
    a.set_title(t)


# Out[32]:

# image file:

# In[75]:

figures = (f1, f2, f3, f4, f5, f6, f7)


# In[76]:

d = '/Users/ath/Desktop/l4'
for i, f in enumerate(figures):
    f.savefig(d + '/fig_' + str(i) + '.png', dpi=100, bbox_inches='tight')

# In[74]:

f = np.ones((5, 5)) / 25
boxed = convolve2d(degraded, f, 'same')
plt.imshow(boxed)
f7 = multi_plot(1, 2, (lee_cam, boxed),
                ('Lee Filter', 'Gaussian Filtered\n$\sigma=30$'), fs=(8, 4))
