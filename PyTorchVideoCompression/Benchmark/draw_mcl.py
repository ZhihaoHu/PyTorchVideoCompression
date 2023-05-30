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

bpp, psnr = [0.183169, 0.112944, 0.079046, 0.055994], [38.473279, 37.360721, 36.239858, 35.001531]
FVC, = plt.plot(bpp, psnr, "c-o", color="dimgrey", linewidth=LineWidth, label='FVC')

bpp, psnr = [0.177528, 0.105804, 0.069346, 0.044944], [38.044085, 36.90718, 35.749727, 34.440707]
rafc, = plt.plot(bpp, psnr, 'c-*', color="blueviolet", linewidth=LineWidth, label='HU_ECCV20')
    
bpp, psnr = [0.211752, 0.133820333, 0.093098, 0.067195667], [38.161138, 37.13298733, 35.91979567, 34.65774567]
LU, = plt.plot(bpp, psnr, "c-o", color="royalblue", linewidth=LineWidth, label='LU_ECCV20')


bpp, psnr =  [0.043773, 0.065584, 0.093487, 0.152313, 0.260900, 0.416239], [33.459643, 34.659647, 36.299582, 37.555222, 38.784670, 39.781974]
EA, = plt.plot(bpp, psnr, "g-o", color="orange", linewidth=LineWidth, label='EA_CVPR20')

bpp, psnr = [0.2156, 0.1203, 0.082, 0.0628], [38.6349, 37.4613, 36.2621, 34.8514]
Liu, = plt.plot(bpp, psnr, "c-o", color="green", linewidth=LineWidth, label='Liu et al.')    

bpp, psnr = [0.031640613666666664, 0.045372268000000014, 0.06589602666666668, 0.09751675000000004, 0.13806897999999998, 0.20986701666666663], [34.08598, 35.20929, 36.23114666666667, 37.16912333333334, 37.964886666666665, 38.84489333333334]
ELF, = plt.plot(bpp, psnr, "y-o", color="gold", linewidth=LineWidth, label='ELF-VC')

bpp, psnr = [0.036842105312160804, 0.055034111193376936, 0.08214686848718189, 0.1305168217933064], [34.16064132372538, 35.47007208400302, 36.663099307484096, 37.73685781055027] 
DCVC, = plt.plot(bpp, psnr, "y-o", color="hotpink", linewidth=LineWidth, label='DCVC')


# bpp = [0.07350666666666665, 0.09998666666666667, 0.14866666666666667, 0.20885666666666666]
# psnr = [34.19015183584638, 34.97325683520819, 36.4162965039923, 37.564191921189945]
# MLVC, = plt.plot(bpp, psnr, "m-o", color='darkorange', linewidth=LineWidth, label='M-LVC')


bpp = [0.067980956, 0.080343472, 0.102782285, 0.133500782, 0.18545014, 0.281805116, 0.376178686]#, 0.543077581]
psnr = [34.33114855, 35.00107532, 35.7351181, 36.56075116, 37.47248012, 38.5945216, 39.43083815]#, 40.29353013]
msssim = [0.956373977, 0.96103441, 0.966706579, 0.971418613, 0.976173806, 0.982212426, 0.984971718]#, 0.987377043]
iccv19, = plt.plot(bpp, psnr, "c-*", linewidth=LineWidth, label='AD_ICCV19')

bpp, psnr, msssim = [0.063445, 0.090866, 0.129456, 0.204172], [34.275199, 35.656182, 36.862051, 38.003749], [0.953543, 0.964906, 0.972667, 0.978116]
DVC, = plt.plot(bpp, psnr, "y-o", linewidth=LineWidth, label='DVC')

bpp, psnr = [0.204917, 0.123134, 0.083819, 0.052188], [39.210012, 38.194799, 37.252419, 36.053591]
C2F, = plt.plot(bpp, psnr, "y-o", color="indigo", linewidth=LineWidth, label='C2F')

# Ours default
# bpp, psnr, msssim = [0.415409871, 0.186455224, 0.096197474, 0.05784043], [38.95635439, 37.12659472, 35.54274685, 33.96921318], [0.983710123, 0.975398139, 0.966838741, 0.956096359]
# h264, = plt.plot(bpp, psnr, "m--s", linewidth=LineWidth, label='H.264')

# bpp = [0.335718128, 0.151147044, 0.075029043, 0.042250736]
# psnr = [38.87289862, 37.21245689, 35.68806736, 34.13927989]
# msssim = [0.981549357, 0.974198804, 0.966348014, 0.956139709]
# h265, = plt.plot(bpp, psnr, "r--v", linewidth=LineWidth, label='H.265')

bpp, psnr, msssim = [0.158163682, 0.083452922, 0.047839759, 0.029251102], [38.53680791, 37.19704719, 35.8682293, 34.66009104], [0.978565985, 0.972857914, 0.966124373, 0.957971806]
HM, = plt.plot(bpp, psnr, "r--v", linewidth=LineWidth, label='HM')

bpp, psnr, msssim = [0.136736442037515, 0.0670341156117102, 0.0374102454419172, 0.0221052901684006], [38.5088890794119, 37.1940123059836, 35.9136068283541, 34.672893890491], [0.97846281349025, 0.972782567801869, 0.966698353858581, 0.958620965561768]
VTM, = plt.plot(bpp, psnr, "b--v", linewidth=LineWidth, label='VTM')


savepathpsnr = prefix + '/MCL_psnr' + '.png'
savepathpsnreps = prefix + '/MCL_psnr' + '.eps'
print(prefix)
if not os.path.exists(prefix):
    os.makedirs(prefix)
plt.legend(handles=[HM, VTM, DVC, iccv19, EA, LU, rafc, Liu, FVC, ELF, DCVC, C2F], loc=4)
plt.grid()
plt.xlabel('Bpp')
plt.ylabel('PSNR')
plt.title('MCL-JCV dataset')
plt.savefig(savepathpsnr)
# plt.savefig(savepathpsnreps, format='eps', dpi=300, bbox_inches='tight')
plt.clf()

# ----------------------------------------MSSSIM-------------------------------------------------

bpp, psnr = [0.327997, 0.205193, 0.122516, 0.078993], [0.988995, 0.985406, 0.980099, 0.973502]
FVC, = plt.plot(bpp, psnr, "c-o", color="dimgrey", linewidth=LineWidth, label='FVC')


bpp, psnr = [0.21493, 0.131528, 0.081933, 0.051843], [0.98331, 0.978267, 0.972588, 0.964108]
rafc, = plt.plot(bpp, psnr, 'c-*', color="blueviolet", linewidth=LineWidth, label='HU_ECCV20')
    
bpp, psnr = [0.211752, 0.133820333, 0.093098, 0.067195667], [0.979231,0.974493667,0.968179333,0.958875667]
LU, = plt.plot(bpp, psnr, "c-o", color="royalblue", linewidth=LineWidth, label='LU_ECCV20')

bpp, psnr =  [0.039060, 0.059494, 0.084072, 0.122407, 0.228456, 0.326567], [0.958350, 0.965656, 0.973480, 0.977267, 0.984815, 0.987433]
EA, = plt.plot(bpp, psnr, "g-o", color="orange", linewidth=LineWidth, label='EA_CVPR20')

bpp, psnr = [0.4664, 0.2951, 0.1752, 0.1074], [0.99009, 0.98783, 0.98307, 0.9774]
Liu, = plt.plot(bpp, psnr, "c-o", color="green", linewidth=LineWidth, label='Liu et al.')

bpp, msssim =  [0.028632991, 0.044951392666666666, 0.07287183333333334, 0.11613923, 0.19002066333333334, 0.30141713333333336], [0.9616781000000003, 0.9696877, 0.9759109333333333, 0.9810244333333332, 0.9856295666666669, 0.9889936000000001]
ELF, = plt.plot(bpp, msssim, "y-o", color="gold", linewidth=LineWidth, label='ELF-VC')

bpp, msssim = [0.061002279768288314, 0.10433220013841572, 0.18810562626776042, 0.2957224117895519], [0.9700077049599753, 0.9777150785923004, 0.9841031651033295, 0.9882810191644562] 
DCVC, = plt.plot(bpp, msssim, "y-o", color="hotpink", linewidth=LineWidth, label='DCVC')

bpp = [0.067980956, 0.080343472, 0.102782285, 0.133500782, 0.18545014, 0.281805116, 0.376178686]#, 0.543077581]
psnr = [34.33114855, 35.00107532, 35.7351181, 36.56075116, 37.47248012, 38.5945216, 39.43083815]#, 40.29353013]
msssim = [0.956373977, 0.96103441, 0.966706579, 0.971418613, 0.976173806, 0.982212426, 0.984971718]#, 0.987377043]
iccv19, = plt.plot(bpp, msssim, "c-*", linewidth=LineWidth, label='AD_ICCV19')

bpp, psnr, msssim = [0.063445, 0.090866, 0.129456, 0.204172], [34.275199, 35.656182, 36.862051, 38.003749], [0.953543, 0.964906, 0.972667, 0.978116]
DVC, = plt.plot(bpp, msssim, "y-o", linewidth=LineWidth, label='DVC')

bpp, msssim = [0.267879, 0.147459, 0.084217, 0.051453], [0.988898, 0.983968, 0.978599, 0.972453]
C2F, = plt.plot(bpp, msssim, "y-o", color="indigo", linewidth=LineWidth, label='C2F')

#default
# bpp, psnr, msssim = [0.415409871, 0.186455224, 0.096197474, 0.05784043], [38.95635439, 37.12659472, 35.54274685, 33.96921318], [0.983710123, 0.975398139, 0.966838741, 0.956096359]
# h264, = plt.plot(bpp, msssim, "m--s", linewidth=LineWidth, label='H.264')

# bpp = [0.335718128, 0.151147044, 0.075029043, 0.042250736]
# psnr = [38.87289862, 37.21245689, 35.68806736, 34.13927989]
# msssim = [0.981549357, 0.974198804, 0.966348014, 0.956139709]
# h265, = plt.plot(bpp, msssim, "r--v", linewidth=LineWidth, label='H.265')

bpp, psnr, msssim = [0.158163682, 0.083452922, 0.047839759, 0.029251102], [38.53680791, 37.19704719, 35.8682293, 34.66009104], [0.978565985, 0.972857914, 0.966124373, 0.957971806]
HM, = plt.plot(bpp, msssim, "r--v", linewidth=LineWidth, label='HM')

bpp, psnr, msssim = [0.136736442037515, 0.0670341156117102, 0.0374102454419172, 0.0221052901684006], [38.5088890794119, 37.1940123059836, 35.9136068283541, 34.672893890491], [0.97846281349025, 0.972782567801869, 0.966698353858581, 0.958620965561768]
VTM, = plt.plot(bpp, msssim, "b--v", linewidth=LineWidth, label='VTM')


savepathmsssim = prefix + '/' + 'MCL_msssim' + '.png'
savepathmsssimeps = prefix + '/' + 'MCL_msssim' + '.eps'
plt.legend(handles=[HM, VTM, DVC, EA, iccv19, LU, rafc, Liu, FVC, ELF, DCVC, C2F], loc=4)
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