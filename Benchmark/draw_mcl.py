import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import imageio
import cv2
import numpy as np

prefix = 'MCLresults'
LineWidth = 2
font = {'family': 'serif', 'weight': 'normal', 'size': 12}
matplotlib.rc('font', **font)

bpp, psnr = [0.177528, 0.105804, 0.069346, 0.044944], [38.044085, 36.90718, 35.749727, 34.440707]
rafc, = plt.plot(bpp, psnr, 'c-*', color="blueviolet", linewidth=LineWidth, label='HU_ECCV20')


bpp, psnr =  [0.043773, 0.065584, 0.093487, 0.152313, 0.260900, 0.416239], [33.459643, 34.659647, 36.299582, 37.555222, 38.784670, 39.781974]
EA, = plt.plot(bpp, psnr, "g-o", color="orange", linewidth=LineWidth, label='EA_CVPR20')


# bpp = [0.07350666666666665, 0.09998666666666667, 0.14866666666666667, 0.20885666666666666]
# psnr = [34.19015183584638, 34.97325683520819, 36.4162965039923, 37.564191921189945]
# MLVC, = plt.plot(bpp, psnr, "m-o", color='darkorange', linewidth=LineWidth, label='M-LVC')


bpp = [0.067980956, 0.080343472, 0.102782285, 0.133500782, 0.18545014, 0.281805116, 0.376178686]#, 0.543077581]
psnr = [34.33114855, 35.00107532, 35.7351181, 36.56075116, 37.47248012, 38.5945216, 39.43083815]#, 40.29353013]
msssim = [0.956373977, 0.96103441, 0.966706579, 0.971418613, 0.976173806, 0.982212426, 0.984971718]#, 0.987377043]
iccv19, = plt.plot(bpp, psnr, "c-*", linewidth=LineWidth, label='AD_ICCV19')

bpp, psnr, msssim = [0.063445, 0.090866, 0.129456, 0.204172], [34.275199, 35.656182, 36.862051, 38.003749], [0.953543, 0.964906, 0.972667, 0.978116]
DVC, = plt.plot(bpp, psnr, "y-o", linewidth=LineWidth, label='DVC')

# Ours default
bpp, psnr, msssim = [0.415409871, 0.186455224, 0.096197474, 0.05784043], [38.95635439, 37.12659472, 35.54274685, 33.96921318], [0.983710123, 0.975398139, 0.966838741, 0.956096359]
h264, = plt.plot(bpp, psnr, "m--s", linewidth=LineWidth, label='H.264')

bpp = [0.335718128, 0.151147044, 0.075029043, 0.042250736]
psnr = [38.87289862, 37.21245689, 35.68806736, 34.13927989]
msssim = [0.981549357, 0.974198804, 0.966348014, 0.956139709]
h265, = plt.plot(bpp, psnr, "r--v", linewidth=LineWidth, label='H.265')


savepathpsnr = prefix + '/MCL_psnr' + '.png'
savepathpsnreps = prefix + '/MCL_psnr' + '.eps'
print(prefix)
if not os.path.exists(prefix):
    os.makedirs(prefix)
plt.legend(handles=[h264, h265, DVC, iccv19, EA, rafc], loc=4)
plt.grid()
plt.xlabel('Bpp')
plt.ylabel('PSNR')
plt.title('MCL-JCV dataset')
plt.savefig(savepathpsnr)
# plt.savefig(savepathpsnreps, format='eps', dpi=300, bbox_inches='tight')
plt.clf()

# ----------------------------------------MSSSIM-------------------------------------------------


bpp, psnr = [0.21493, 0.131528, 0.081933, 0.051843], [0.98331, 0.978267, 0.972588, 0.964108]
rafc, = plt.plot(bpp, psnr, 'c-*', color="blueviolet", linewidth=LineWidth, label='HU_ECCV20')

bpp, psnr =  [0.039060, 0.059494, 0.084072, 0.122407, 0.228456, 0.326567], [0.958350, 0.965656, 0.973480, 0.977267, 0.984815, 0.987433]
EA, = plt.plot(bpp, psnr, "g-o", color="orange", linewidth=LineWidth, label='EA_CVPR20')

bpp = [0.067980956, 0.080343472, 0.102782285, 0.133500782, 0.18545014, 0.281805116, 0.376178686]#, 0.543077581]
psnr = [34.33114855, 35.00107532, 35.7351181, 36.56075116, 37.47248012, 38.5945216, 39.43083815]#, 40.29353013]
msssim = [0.956373977, 0.96103441, 0.966706579, 0.971418613, 0.976173806, 0.982212426, 0.984971718]#, 0.987377043]
iccv19, = plt.plot(bpp, msssim, "c-*", linewidth=LineWidth, label='AD_ICCV19')

bpp, psnr, msssim = [0.063445, 0.090866, 0.129456, 0.204172], [34.275199, 35.656182, 36.862051, 38.003749], [0.953543, 0.964906, 0.972667, 0.978116]
DVC, = plt.plot(bpp, msssim, "y-o", linewidth=LineWidth, label='DVC')

#default
bpp, psnr, msssim = [0.415409871, 0.186455224, 0.096197474, 0.05784043], [38.95635439, 37.12659472, 35.54274685, 33.96921318], [0.983710123, 0.975398139, 0.966838741, 0.956096359]
h264, = plt.plot(bpp, msssim, "m--s", linewidth=LineWidth, label='H.264')

bpp = [0.335718128, 0.151147044, 0.075029043, 0.042250736]
psnr = [38.87289862, 37.21245689, 35.68806736, 34.13927989]
msssim = [0.981549357, 0.974198804, 0.966348014, 0.956139709]
h265, = plt.plot(bpp, msssim, "r--v", linewidth=LineWidth, label='H.265')


savepathmsssim = prefix + '/' + 'MCL_msssim' + '.png'
savepathmsssimeps = prefix + '/' + 'MCL_msssim' + '.eps'
plt.legend(handles=[h264, h265, DVC, EA, iccv19, rafc], loc=4)
plt.grid()
plt.xlabel('Bpp')
plt.ylabel('MS-SSIM')
plt.title('MCL-JCV dataset')
plt.savefig(savepathmsssim)
# plt.savefig(savepathmsssimeps, format='eps', dpi=300, bbox_inches='tight')
plt.clf()


savepath = prefix + '/' + 'MCL' + '.png'
img1 = cv2.imread(savepathpsnr)
img2 = cv2.imread(savepathmsssim)

image = np.concatenate((img1, img2), axis=1)
cv2.imwrite(savepath, image)