import numpy
import math
import sys
import numpy
from scipy import signal
from scipy import ndimage
from skimage import io
import sys
import re
import math
import sys
import numpy
from scipy import signal
from scipy import ndimage

# import gauss
import matplotlib.pyplot as plt

#!/usr/bin/env python
"""Module providing functionality surrounding gaussian function.
"""
SVN_REVISION = '$LastChangedRevision: 16541 $'

import sys
import numpy

# def gaussian2(size, sigma):
#     """Returns a normalized circularly symmetric 2D gauss kernel array

#     f(x,y) = A.e^{-(x^2/2*sigma^2 + y^2/2*sigma^2)} where

#     A = 1/(2*pi*sigma^2)

#     as define by Wolfram Mathworld
#     http://mathworld.wolfram.com/GaussianFunction.html
#     """
#     A = 1/(2.0*numpy.pi*sigma**2)
#     x, y = numpy.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
#     g = A*numpy.exp(-((x**2/(2.0*sigma**2))+(y**2/(2.0*sigma**2))))
#     return g

def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = numpy.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = numpy.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()


def ssim(img1, img2, cs_map=False):
    """Return the Structural Similarity Map corresponding to input images img1
    and img2 (images are assumed to be uint8)

    This function attempts to mimic precisely the functionality of ssim.m a
    MATLAB provided by the author's of SSIM
    https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
    """
    img1 = img1.astype(numpy.float64)
    img2 = img2.astype(numpy.float64)
    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 255  #bitdepth of image
    C1 = (K1 * L)**2
    C2 = (K2 * L)**2
    mu1 = signal.fftconvolve(window, img1, mode='valid')
    mu2 = signal.fftconvolve(window, img2, mode='valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = signal.fftconvolve(window, img1 * img1, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(window, img2 * img2, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(window, img1 * img2, mode='valid') - mu1_mu2
    if cs_map:
        return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)),
                (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))


def msssim(img1, img2):
    """This function implements Multi-Scale Structural Similarity (MSSSIM) Image
    Quality Assessment according to Z. Wang's "Multi-scale structural similarity
    for image quality assessment" Invited Paper, IEEE Asilomar Conference on
    Signals, Systems and Computers, Nov. 2003

    Author's MATLAB implementation:-
    http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
    """
    level = 5
    weight = numpy.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    downsample_filter = numpy.ones((2, 2)) / 4.0
    im1 = img1.astype(numpy.float64)
    im2 = img2.astype(numpy.float64)
    mssim = numpy.array([])
    mcs = numpy.array([])
    for l in range(level):
        ssim_map, cs_map = ssim(im1, im2, cs_map=True)
        mssim = numpy.append(mssim, ssim_map.mean())
        mcs = numpy.append(mcs, cs_map.mean())
        filtered_im1 = ndimage.filters.convolve(im1, downsample_filter, mode='reflect')
        filtered_im2 = ndimage.filters.convolve(im2, downsample_filter, mode='reflect')
        im1 = filtered_im1[::2, ::2]
        im2 = filtered_im2[::2, ::2]
    return (numpy.prod(mcs[0:level - 1]**weight[0:level - 1]) * (mssim[level - 1]**weight[level - 1]))


def psnr(ref, target):
    diff = ref/255.0 - target/255.0
    diff = diff.flatten('C')
    rmse = math.sqrt(numpy.mean(diff**2.))
    return 20 * math.log10(1.0 / (rmse))


def main():
    im_width = float(sys.argv[2])
    im_height = float(sys.argv[3])

    prefix = 'out/'
    psnr_arr = []
    msssim_arr = []
    bpp_arr = []

    with open('ffreport.log') as f:
        lines = f.readlines()

    size_line = []
    for l in lines:
        if ", size " in l:
            size = l.split(',')[1]
            size_line.append(int(size[5:]))

    size_line = numpy.array(size_line)*8.0/(im_width*im_height)
    # print(size_line)

    # bpp_str = ''
    # for l in lines:
    #     if "Lsize" in l:
    #         bpp_str = l
    #         break

    # bpp_strs = re.findall(r"[-+]?\d*\.\d+|\d+", bpp_str)
    # bpp = float(bpp_strs[3])*8000/(im_width*im_height)

    import time
    for i in range(len(size_line)):
#
        if (i) % 12 == 0:
        #if True:
            source = prefix + 'source/img' + "{0:0=6d}".format(i+1) + '.png'
            h265 = prefix + 'h265/img' + "{0:0=6d}".format(i+1) + '.png'

            source_img = io.imread(source)
            h265_img = io.imread(h265)

            # print(source_img.shape)
            # print(h265_img.shape)

            psnr_val = 0#psnr(source_img, h265_img)
            tmpssim = 0
            # tmpssim = msssim(h265_img[:, :, 0], source_img[:, :, 0])
            # tmpssim += msssim(h265_img[:, :, 1], source_img[:, :, 1])
            # tmpssim += msssim(h265_img[:, :, 2], source_img[:, :, 2])
            ms_ssim_val = tmpssim/3.0

            psnr_arr.append(psnr_val)
            msssim_arr.append(ms_ssim_val)
            bpp_arr.append(size_line[i])

            # print(psnr_val)
            # print(ms_ssim_val)
            # print(size_line[i])

    print(sys.argv[1])

    print('psnr:' +  str(numpy.array(psnr_arr).mean(0)))
    print('bpp:' +str(numpy.array(bpp_arr).mean(0)))
    print('msssim:' +str(numpy.array(msssim_arr).mean(0)))

    # plt.plot(bpp, msssim_arr)
    # plt.ylabel('MS-SSIM')
    # plt.xlabel('bpp')
    # plt.show()

    # plt.plot(bpp_arr, psnr_arr, 'ro')
    # plt.ylabel('PSNR')
    # plt.xlabel('bpp')
    # plt.show()

    # plt.plot(bpp_arr, 'ro')
    # plt.ylabel('BPP')
    # plt.title(sys.argv[1])
    # plt.savefig(sys.argv[1] + 'bpp.png')
    # plt.show()

    # plt.plot(psnr_arr, 'ro')
    # plt.ylabel('PSNR')
    # plt.title(sys.argv[1])
    # plt.savefig(sys.argv[1] + 'psnr.png')
    # plt.show()

    # i_f = []
    # p_f = []

    # for i in range(len(bpp_arr)):
    #     if i % 10 == 0:
    #         i_f.append(bpp_arr[i])
    #     else:
    #         p_f.append(bpp_arr[i])

    # print(sys.argv[1] + ': I vs P = '+ str(sum(i_f)/len(i_f)) + " : " + str(sum(p_f)/len(p_f)))


if __name__ == '__main__':
    sys.exit(main())
