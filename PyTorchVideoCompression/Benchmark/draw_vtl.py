import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import imageio
import cv2
import numpy as np

prefix = 'VTLresults'
LineWidth = 2
font = {'family': 'serif', 'weight': 'normal', 'size': 12}
matplotlib.rc('font', **font)

bpp, psnr = [0.306566, 0.195756, 0.133961, 0.092266], [35.019422, 33.60735, 32.281443, 30.989942]
FVC, = plt.plot(bpp, psnr, "c-o", color="dimgrey", linewidth=LineWidth, label='FVC')


bpp, psnr = [0.296787, 0.19001, 0.126534, 0.080851], [34.303688, 33.036154, 31.737594, 30.490966]
rafc, = plt.plot(bpp, psnr, 'c-*', color="blueviolet", linewidth=LineWidth, label='HU_ECCV20')

bpp, psnr = [0.3059625, 0.199, 0.136516, 0.093883], [34.5241105,  33.20, 31.7834635, 30.4204885]
LU, = plt.plot(bpp, psnr, "c-o", color="royalblue", linewidth=LineWidth, label='LU_ECCV20')

bpp, psnr = [0.3129, 0.185, 0.1033, 0.07], [35.2057, 33.7928, 32.2786, 30.8637]
Liu, = plt.plot(bpp, psnr, "c-o", color="green", linewidth=LineWidth, label='Liu et al.')

bpp = [0.149002168, 0.194118677, 0.269459397, 0.484332918]#, 0.956521083]
psnr = [31.67178903, 32.56079993, 33.57877741, 35.89857494]#, 38.44629636]
iccv, = plt.plot(bpp, psnr, "c-*", linewidth=LineWidth, label='AD_ICCV19')

bpp, psnr, msssim = [0.317761, 0.207319, 0.143094, 0.097783], [34.326297, 33.068078, 31.720454, 30.447007], [0.982151, 0.977638, 0.971062, 0.961707]
DVC, = plt.plot(bpp, psnr, "y-o", linewidth=LineWidth, label='DVC')

# Ours default
bpp, psnr, msssim = [0.560790221, 0.296493945, 0.163039168, 0.096774032], [35.96539139, 33.47594351, 31.24127318, 29.19766722], [0.989001595, 0.981267207, 0.971202069, 0.958924346]
h264, = plt.plot(bpp, psnr, "m--s", linewidth=LineWidth, label='H.264')
bpp, psnr, msssim = [0.504603189, 0.270359303, 0.144005192, 0.0822622], [35.93173383, 33.57071872, 31.32046883, 29.25416181], [0.988627673, 0.982025883, 0.971317029, 0.957919182] # my
h265, = plt.plot(bpp, psnr, "r--v", linewidth=LineWidth, label='H.265')

savepathpsnr = prefix + '/VTL_psnr'# + '.eps'
print(prefix)
if not os.path.exists(prefix):
    os.makedirs(prefix)
plt.legend(handles=[h264, h265, DVC, iccv, LU, rafc, Liu, FVC], loc=4)
plt.grid()
plt.xlabel('Bpp')
plt.ylabel('PSNR')
plt.title('VTL dataset')
# plt.savefig(savepathpsnr + '.eps', format='eps', dpi=300, bbox_inches='tight')
plt.savefig(savepathpsnr + '.png')
plt.clf()

# ----------------------------------------MSSSIM-------------------------------------------------

# bpp = [0.3113, 0.2699, 0.2290, 0.1983, 0.16239, 0.12377]
# msssim = [0.9718572754, 0.9711699173, 0.96866567, 0.9670768429, 0.9659088446, 0.961264922]
# eccv, = plt.plot(bpp, msssim, "b-*", linewidth=LineWidth, label='Wu_ECCV')

bpp, psnr = [0.292538, 0.189176, 0.113552, 0.072098], [0.989218, 0.984481, 0.97661, 0.966557]
FVC, = plt.plot(bpp, psnr, "c-o", color="dimgrey", linewidth=LineWidth, label='FVC')

bpp, msssim = [0.291837, 0.189759, 0.121654, 0.070671], [0.984102, 0.978341, 0.97124, 0.959966]
rafc, = plt.plot(bpp, msssim, 'c-*', color="blueviolet", linewidth=LineWidth, label='HU_ECCV20')

bpp, psnr = [0.3059625, 0.199, 0.136516, 0.093883], [0.9838005, 0.9793955, 0.9732655, 0.964076]
LU, = plt.plot(bpp, psnr, "c-o", color="royalblue", linewidth=LineWidth, label='LU_ECCV20')

bpp, psnr = [0.3449, 0.2146, 0.1282, 0.0777], [0.99114, 0.98771, 0.981, 0.97375]
Liu, = plt.plot(bpp, psnr, "c-o", color="green", linewidth=LineWidth, label='Liu et al.')

bpp = [0.149002168, 0.194118677, 0.269459397, 0.484332918]#, 0.956521083]
psnr = [0.971382803, 0.975893586, 0.980153643, 0.988934007]#, 0.994301121]
iccv, = plt.plot(bpp, psnr, "c-*", linewidth=LineWidth, label='AD_ICCV19')

bpp, psnr, msssim = [0.317761, 0.207319, 0.143094, 0.097783], [34.326297, 33.068078, 31.720454, 30.447007], [0.982151, 0.977638, 0.971062, 0.961707]
DVC, = plt.plot(bpp, msssim, "y-o", linewidth=LineWidth, label='DVC')



bpp, psnr, msssim = [0.560790221, 0.296493945, 0.163039168, 0.096774032], [35.96539139, 33.47594351, 31.24127318, 29.19766722], [0.989001595, 0.981267207, 0.971202069, 0.958924346]
h264, = plt.plot(bpp, msssim, "m--s", linewidth=LineWidth, label='H.264')

bpp, psnr, msssim = [0.504603189, 0.270359303, 0.144005192, 0.0822622], [35.93173383, 33.57071872, 31.32046883, 29.25416181], [0.988627673, 0.982025883, 0.971317029, 0.957919182]#my
h265, = plt.plot(bpp, msssim, "r--v", linewidth=LineWidth, label='H.265')


savepathmsssim = prefix + '/' + 'VTL_msssim'# + '.eps'
plt.legend(handles=[h264, h265, DVC, iccv, LU, rafc, Liu, FVC], loc=4)
plt.grid()
plt.xlabel('Bpp')
plt.ylabel('MS-SSIM')
plt.title('VTL dataset')
# plt.savefig(savepathmsssim + '.eps', format='eps', dpi=300, bbox_inches='tight')
plt.savefig(savepathmsssim + '.png')
plt.clf()


savepath = prefix + '/' + 'VTL' + '.png'
img1 = cv2.imread(savepathpsnr + '.png')
img2 = cv2.imread(savepathmsssim + '.png')

image = np.concatenate((img1, img2), axis=1)
cv2.imwrite(savepath, image)