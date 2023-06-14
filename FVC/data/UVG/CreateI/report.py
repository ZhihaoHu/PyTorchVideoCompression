import numpy as np

with open('result.txt') as f:
    lines = f.readlines()

psnr = []
bpp = []
msssim = []

for l in lines:             
    if "psnr" in l and l[5:8] != 'nan':
        psnr.append(float(l[5:]))
    if "bpp" in l and l[4:7] != 'nan':
        bpp.append(float(l[4:]))
    if "msssim" in l and l[7:10] != 'nan':
        msssim.append(float(l[7:]))

# print('final psnr')
# print(np.array(psnr).mean(0))
print('final bpp')
print(np.array(bpp).mean(0))
# print('final mssim')
# print(np.array(msssim).mean(0))