import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import imageio
import cv2
import numpy as np

def drawhevca():
    prefix = 'HEVCresults'
    font = {'family': 'serif', 'weight': 'normal', 'size': 12}
    matplotlib.rc('font', **font)
    LineWidth = 2

    bpp, psnr = [0.252517, 0.172729, 0.115206, 0.087225], [36.445944, 35.251347, 33.906883, 32.385862]
    RaFC, = plt.plot(bpp, psnr, "c-o", color="blueviolet", linewidth=LineWidth, label='HU_ECCV20')
    
    bpp = [0.242736, 0.167867, 0.117996, 0.083094]
    psnr = [36.054912, 34.835034, 33.490675, 32.040395]
    DVC, = plt.plot(bpp, psnr, "y-o", linewidth=LineWidth, label='DVC')

    bpp = [0.34925124, 0.188116367, 0.112805635, 0.072584189]
    psnr = [36.0594951, 34.19014687, 32.43235565, 30.67496279]
    msssim = [0.984545015, 0.978037778, 0.969541362, 0.956833816]
    h264, = plt.plot(bpp, psnr, "m--s", linewidth=LineWidth, label='H.264')

    bpp = [0.317619209, 0.169611445, 0.098485186, 0.059770479]
    psnr = [36.11257808, 34.30115064, 32.57508237, 30.79521459]
    msssim = [0.98365547, 0.977146255, 0.968649611, 0.956017076]
    h265, = plt.plot(bpp, psnr, "r--v", linewidth=LineWidth, label='H.265')

    savepathpsnr = prefix + '/' + 'HEVCClass_A_psnr'# + '.eps'
    print(prefix)
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    plt.legend(handles=[h264, h265, DVC, RaFC], loc=4)

    plt.grid()
    plt.xlabel('Bpp')
    plt.ylabel('PSNR')
    plt.title('HEVC Class A dataset')
    # plt.savefig(savepathpsnr + '.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.savefig(savepathpsnr + '.png')
    plt.clf()

    # ----------------------------------------MSSSIM-------------------------------------------------
    bpp, msssim = [0.302868, 0.194612, 0.128278, 0.075894], [0.986527, 0.981522, 0.97537, 0.966448]
    RaFC, = plt.plot(bpp, msssim, "c-o", color="blueviolet", linewidth=LineWidth, label='HU_ECCV20')

    msssim = [0.98218, 0.977847, 0.971829, 0.962887]
    bpp = [0.242736, 0.167867, 0.117996, 0.083094]
    DVC, = plt.plot(bpp, msssim, "y-p", linewidth=LineWidth, label='DVC')

    bpp = [0.34925124, 0.188116367, 0.112805635, 0.072584189]
    psnr = [36.0594951, 34.19014687, 32.43235565, 30.67496279]
    msssim = [0.984545015, 0.978037778, 0.969541362, 0.956833816]
    h264, = plt.plot(bpp, msssim, "m--s", linewidth=LineWidth, label='H.264')

    bpp = [0.317619209, 0.169611445, 0.098485186, 0.059770479]
    psnr = [36.11257808, 34.30115064, 32.57508237, 30.79521459]
    msssim = [0.98365547, 0.977146255, 0.968649611, 0.956017076]
    h265, = plt.plot(bpp, msssim, "r--v", linewidth=LineWidth, label='H.265')

    savepathmsssim = prefix + '/' + 'HEVCClass_A_msssim'# + '.eps'
    
    plt.legend(handles=[h264, h265, DVC, RaFC], loc=4)

    plt.grid()
    plt.xlabel('Bpp')
    plt.ylabel('MS-SSIM')
    plt.title('HEVC Class A dataset')
    # plt.savefig(savepathmsssim + '.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.savefig(savepathmsssim + '.png')
    plt.clf()

    savepath = prefix + '/' + 'HEVCClass_A.png'
    img1 = cv2.imread(savepathpsnr + '.png')
    img2 = cv2.imread(savepathmsssim + '.png')

    image = np.concatenate((img1, img2), axis=1)
    cv2.imwrite(savepath, image)

drawhevca()