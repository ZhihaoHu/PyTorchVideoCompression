import matplotlib
from numpy.lib.npyio import save
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import cv2

def todb(x):
    return x

def hevcdrawplt(name, lbpp, lpsnr, lmsssim, tb, global_step, la='new', testfull=False):
    prefix = 'performance'
    if testfull:
        prefix = 'fullpreformance'
    LineWidth = 2
    test, = plt.plot(lbpp, lpsnr, marker='x', color='k', linewidth=LineWidth, label=la)

    savename = name.replace(" ", "_")

    if name == 'HEVC Class B dataset':

        bpp, psnr = [0.269053, 0.146596, 0.097074, 0.067416], [35.391955, 34.391431, 33.370759, 32.255203]
        FVC, = plt.plot(bpp, psnr, "g-o", linewidth=LineWidth, label='FVC')

        bpp = [0.251340112, 0.114954761, 0.060032682, 0.035127775]
        psnr = [35.66700977, 34.40308441, 33.21724712, 32.13595722]
        HM, = plt.plot(bpp, psnr, "r-v", linewidth=LineWidth, label='HM')
    elif name == 'HEVC Class C dataset':

        bpp, psnr = [0.350834, 0.247527, 0.174569, 0.121681], [33.482413, 32.367214, 31.096299, 29.633528]
        FVC, = plt.plot(bpp, psnr, "g-o", linewidth=LineWidth, label='FVC')

        bpp = [0.395451633, 0.233026324, 0.13980179, 0.086984085]
        psnr = [35.20042775, 33.37764229, 31.5954333, 30.06819418]
        HM, = plt.plot(bpp, psnr, "r-v", linewidth=LineWidth, label='HM')
    elif name == 'HEVC Class D dataset':

        bpp, psnr = [0.382361, 0.275732, 0.195583, 0.136572], [33.687894, 32.427037, 31.009378, 29.455373]
        FVC, = plt.plot(bpp, psnr, "g-o", linewidth=LineWidth, label='FVC')

        bpp = [0.421776, 0.255772569, 0.154388021, 0.096538357]
        psnr = [35.30837148, 33.31139678, 31.35738521, 29.71199453]
        HM, = plt.plot(bpp, psnr, "r-v", linewidth=LineWidth, label='HM')
    elif name == 'HEVC Class E dataset':

        bpp, psnr = [0.104291, 0.06196, 0.046346, 0.034603], [40.031664, 38.871796, 37.75919, 36.411769]
        FVC, = plt.plot(bpp, psnr, "g-o", linewidth=LineWidth, label='FVC')

        bpp = [0.055168649, 0.026553829, 0.015017815, 0.00944111]
        psnr = [39.49582704, 38.35633814, 37.15662317, 35.97421709]
        HM, = plt.plot(bpp, psnr, "r-v", linewidth=LineWidth, label='HM')
    else:
        print('no such class : ', name)
        exit(0)
    savepathpsnr = os.path.join(prefix, savename + "_psnr.png")
    print(prefix)
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    plt.legend(handles=[HM, FVC, test], loc=4)
    plt.grid()
    plt.xlabel('Bpp')
    plt.ylabel('PSNR')
    plt.title(name)
    plt.savefig(savepathpsnr)
    plt.clf()

# ----------------------------------------MSSSIM-------------------------------------------------
    test, = plt.plot(lbpp, todb(lmsssim), marker='*', color='black', linewidth=LineWidth, label=la)

    if name == 'HEVC Class B dataset':
        bpp, msssim = [0.443496, 0.261679, 0.142189, 0.08852], todb([0.98591, 0.98055, 0.972843, 0.964454])
        FVC, = plt.plot(bpp, msssim, "g-o", linewidth=LineWidth, label='FVC')

        bpp = [0.251340112, 0.114954761, 0.060032682, 0.035127775]
        psnr = todb([0.972090578, 0.965651358, 0.957313066, 0.947492554])
        HM, = plt.plot(bpp, psnr, "r-v", linewidth=LineWidth, label='HM')

    elif name == 'HEVC Class C dataset':

        bpp, msssim = [0.375906, 0.248239, 0.162061, 0.112118], todb([0.987824, 0.982816, 0.975108, 0.964439])
        FVC, = plt.plot(bpp, msssim, "g-o", linewidth=LineWidth, label='FVC')

        bpp = [0.395451633, 0.233026324, 0.13980179, 0.086984085]
        psnr = todb([0.983137322, 0.976583548, 0.966985639, 0.9554458])
        HM, = plt.plot(bpp, psnr, "r-v", linewidth=LineWidth, label='HM')
    elif name == 'HEVC Class D dataset':
        bpp, msssim = [0.309253, 0.210086, 0.139261, 0.097161], todb([0.989619, 0.984708, 0.976946, 0.966873])
        FVC, = plt.plot(bpp, msssim, "g-o", linewidth=LineWidth, label='FVC')

        bpp = [0.421776, 0.255772569, 0.154388021, 0.096538357]
        psnr = todb([0.986564272, 0.980657509, 0.971165502, 0.959300525])
        HM, = plt.plot(bpp, psnr, "r-v", linewidth=LineWidth, label='HM')
    elif name == 'HEVC Class E dataset':
        bpp, msssim = [0.175973, 0.093687, 0.052362, 0.035589], todb([0.992617, 0.989891, 0.986464, 0.982571])
        FVC, = plt.plot(bpp, msssim, "g-o", linewidth=LineWidth, label='FVC')

        bpp = [0.055168649, 0.026553829, 0.015017815, 0.00944111]
        psnr = todb([0.985909003, 0.983362632, 0.979828185, 0.975380472])
        HM, = plt.plot(bpp, psnr, "r-v", linewidth=LineWidth, label='HM')
    else:
        print('no such class : ', name)
        exit(0)


    savepathmsssim = os.path.join(prefix, savename + "_msssim.png")
    plt.legend(handles=[HM, FVC, test], loc=4)
    plt.grid()
    plt.xlabel('Bpp')
    plt.ylabel('MS-SSIM')
    plt.title(name)
    plt.savefig(savepathmsssim)
    plt.clf()
    
    if tb != None:
        tb.add_image(name + " PSNR", cv2.imread(savepathpsnr).transpose(2,0,1), global_step)
        tb.add_image(name + " MS-SSIM", cv2.imread(savepathmsssim).transpose(2,0,1), global_step)

if __name__ == '__main__':
    labelname = ''
    hevcdrawplt('B', [], [], [], None, 0, la=labelname, testfull=True)
    hevcdrawplt('C', [], [], [], None, 0, la=labelname, testfull=True)
    hevcdrawplt('D', [], [], [], None, 0, la=labelname, testfull=True)
    hevcdrawplt('E', [], [], [], None, 0, la=labelname, testfull=True)