import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import imageio
import cv2

def uvgdrawplt(lbpp, lpsnr, lmsssim, global_step, la='new', testfull=False):
    prefix = 'performance'
    if testfull:
        prefix = 'fullpreformance'
    LineWidth = 2
    test, = plt.plot(lbpp, lpsnr, marker='x', color='black', linewidth=LineWidth, label=la)

    bpp, psnr, msssim = [0.176552, 0.107806, 0.074686, 0.052697], [37.754576, 36.680327, 35.602740, 34.276196], [0.970477, 0.963935, 0.955738, 0.942226]
    baseline, = plt.plot(bpp, psnr, "b-*", linewidth=LineWidth, label='baseline')

    # Ours very fast
    bpp, psnr, msssim = [0.187701631, 0.122491399, 0.084205003, 0.046558501], [36.52492847, 35.78201761, 35.05371763, 33.56996097], [0.968154218, 0.962246563, 0.956369263, 0.942897242]
    h264, = plt.plot(bpp, psnr, "m--s", linewidth=LineWidth, label='H.264')

    bpp, psnr = [0.165663191, 0.109789007, 0.074090183, 0.039677747], [37.29259129, 36.5842637, 35.88754734, 34.46536633]
    h265, = plt.plot(bpp, psnr, "r--v", linewidth=LineWidth, label='H.265')
    
    savepathpsnr = prefix + '/UVG_psnr' + '.png'
    print(prefix)
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    plt.legend(handles=[h264, h265, baseline, test], loc=4)
    plt.grid()
    plt.xlabel('Bpp')
    plt.ylabel('PSNR')
    plt.title('UVG dataset')
    plt.savefig(savepathpsnr)
    plt.clf()

# ----------------------------------------MSSSIM-------------------------------------------------
    test, = plt.plot(lbpp, lmsssim, marker='x', color='black', linewidth=LineWidth, label=la)

    bpp, psnr, msssim = [0.176552, 0.107806, 0.074686, 0.052697], [37.754576, 36.680327, 35.602740, 34.276196], [0.970477, 0.963935, 0.955738, 0.942226]
    baseline, = plt.plot(bpp, msssim, "b-*", linewidth=LineWidth, label='baseline')

    # Ours very fast
    bpp, psnr, msssim = [0.187701631, 0.122491399, 0.084205003, 0.046558501], [36.52492847, 35.78201761, 35.05371763, 33.56996097], [0.968154218, 0.962246563, 0.956369263, 0.942897242]
    h264, = plt.plot(bpp, msssim, "m--s", linewidth=LineWidth, label='H.264')

    bpp, msssim = [0.165663191, 0.074090183, 0.039677747], [0.970470131, 0.960598164, 0.950199185]
    h265, = plt.plot(bpp, msssim, "r--v", linewidth=LineWidth, label='H.265')

    savepathmsssim = prefix + '/' + 'UVG_msssim' + '.png'
    plt.legend(handles=[h264, h265, baseline, test], loc=4)
    plt.grid()
    plt.xlabel('Bpp')
    plt.ylabel('MS-SSIM')
    plt.title('UVG dataset')
    plt.savefig(savepathmsssim)
    plt.clf()

if __name__ == '__main__':
    labelname = ''
    uvgdrawplt([], [], [], 0, la=labelname, testfull=True)