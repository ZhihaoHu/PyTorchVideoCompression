#!/Library/Frameworks/Python.framework/Versions/3.5/bin/python3.5
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime
import json
import math
import time
from six.moves import xrange
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import imageio
import datetime
from .flowlib import flow_to_image
from torch.nn import init

out_channel_N = 64
out_channel_M = 64
out_channel_resN = 128
out_channel_resM = 128
out_channel_mv = 128


def Var(x):
    return Variable(x.cuda())

    
        
def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def Padit(x):
    if x.shape[2] % 2 == 1 or x.shape[3] % 2 == 1:
        return F.pad(x, (0, 2 - (x.shape[3] % 2), 0, 2 - (x.shape[2] % 2)), mode='replicate')
    return x
    
def CalcuPSNR(target, ref):
    diff = ref - target
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff**2.))
    if rmse > 0.:
        return 20 * math.log10(1.0 / (rmse))
    else:
        return 100.

def MSE2PSNR(MSE):
    if MSE > 0:
        return 10 * math.log10(1.0 / (MSE))
    else:
        return 100

def geti(lamb):
    if lamb == 8192:
        return 'BPGQ20'
    elif lamb == 4096:
        return 'BPGQ22'
    elif lamb == 2048:
        return 'BPGQ24'
    elif lamb == 1024:
        return 'BPGQ26'
    elif lamb == 512:
        return 'BPGQ28'
    elif lamb == 256:
        return 'BPGQ30'
    else:
        print("cannot find lambda : %d"%(lamb))
        exit(0)


def Q(x, training):
    if training:
        return x + torch.nn.init.uniform_(torch.zeros_like(x), -0.5, 0.5)
    else:
        return torch.round(x)

def Normit(a, b):
    out = []
    for i in range(a.shape[0]):
        ia = a[i, :, :, :]
        ib = b[i, :, :, :]
        ia[0] = ia[0, :, :] - torch.mean(ia[0, :, :]) + torch.mean(ib[0, :, :])
        ia[1] = ia[1, :, :] - torch.mean(ia[1, :, :]) + torch.mean(ib[1, :, :])
        ia[2] = ia[2, :, :] - torch.mean(ia[2, :, :]) + torch.mean(ib[2, :, :])
        out.append(ia.unsqueeze(0))
    return torch.cat(out, 0)

class ResBlock(nn.Module):
    def __init__(self, inputchannel, outputchannel, kernel_size, stride=1):
        super(ResBlock, self).__init__()
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(inputchannel, outputchannel, kernel_size, stride, padding=kernel_size//2)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(outputchannel, outputchannel, kernel_size, stride, padding=kernel_size//2)
        if inputchannel != outputchannel:
            self.adapt_conv = nn.Conv2d(inputchannel, outputchannel, 1)
        else:
            self.adapt_conv = None

    def forward(self, x):
        x_1 = self.relu1(x)
        firstlayer = self.conv1(x_1)
        firstlayer = self.relu2(firstlayer)
        seclayer = self.conv2(firstlayer)
        if self.adapt_conv is None:
            return x + seclayer
        else:
            return self.adapt_conv(x) + seclayer


class LkResBlock(nn.Module):
    def __init__(self, inputchannel, outputchannel, kernel_size, stride=1):
        super(LkResBlock, self).__init__()
        self.relu1 = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(inputchannel, outputchannel, kernel_size, stride, padding=kernel_size//2)
        self.relu2 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(outputchannel, outputchannel, kernel_size, stride, padding=kernel_size//2)
        if inputchannel != outputchannel:
            self.adapt_conv = nn.Conv2d(inputchannel, outputchannel, 1)
        else:
            self.adapt_conv = None

    def forward(self, x):
        x_1 = self.relu1(x)
        firstlayer = self.conv1(x_1)
        firstlayer = self.relu2(firstlayer)
        seclayer = self.conv2(firstlayer)
        if self.adapt_conv is None:
            return x + seclayer
        else:
            return self.adapt_conv(x) + seclayer


def bilinearupsacling2(inputfeature):
    inputheight = inputfeature.size()[2]
    inputwidth = inputfeature.size()[3]
    outfeature = F.interpolate(inputfeature, (inputheight * 2, inputwidth * 2), mode='bilinear', align_corners=True)
    return outfeature

class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=128, ks=3):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, ks, 1, ks//2, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, ks, 1, ks//2, bias=True)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return identity + out

class Resblocks(nn.Module):

    def __init__(self, nf=128, ks=3):
        super(Resblocks, self).__init__()
        self.res1 = ResidualBlock_noBN(nf, ks)
        self.res2 = ResidualBlock_noBN(nf, ks)
        self.res3 = ResidualBlock_noBN(nf, ks)

    def forward(self, x):
        return x + self.res3(self.res2(self.res1(x)))


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


def same_padding(kernel_size):
    # assumming stride 1
    if isinstance(kernel_size, int):
        return kernel_size // 2
    else:
        return (kernel_size[0] // 2, kernel_size[1] // 2)

class MaskedConvolution2D(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
            *args, mask='A', vertical=False, mask_mode="noblind", **kwargs):
        if "padding" not in kwargs:
            assert "stride" not in kwargs
            kwargs["padding"] = same_padding(kernel_size)
        remove = {"conditional_features", "conditional_image_channels"}
        for feature in remove:
            if feature in kwargs:
                del kwargs[feature]
        super(MaskedConvolution2D, self).__init__(in_channels,
                out_channels, kernel_size, *args, **kwargs)
        Cout, Cin, kh, kw = self.weight.size()
        pre_mask = np.array(np.ones_like(self.weight.data.cpu().numpy())).astype(np.float32)
        yc, xc = kh // 2, kw // 2

        assert mask_mode in {"noblind", "turukin", "fig1-van-den-oord"}
        if mask_mode == "noblind":
            # context masking - subsequent pixels won't hav access
            # to next pixels (spatial dim)
            if vertical:
                if mask == 'A':
                    # In the first layer, can ONLY access pixels above it
                    pre_mask[:, :, yc:, :] = 0.0
                else:
                    # In the second layer, can access pixels above or even with it.
                    # Reason being that the pixels to the right or left of the current pixel
                    #  only have a receptive field of the layer above the current layer and up.
                    pre_mask[:, :, yc+1:, :] = 0.0
            else:
                # All rows after center must be zero
                pre_mask[:, :, yc+1:, :] = 0.0
                ### All rows before center must be zero # XXX: not actually necessary
                ##pre_mask[:, :, :yc, :] = 0.0
                # All columns after center in center row must be zero
                pre_mask[:, :, yc, xc+1:] = 0.0

            if mask == 'A':
                # Center must be zero in first layer
                pre_mask[:, :, yc, xc] = 0.0
            # same pixel masking - pixel won't access next color (conv filter dim)
            #def bmask(i_out, i_in):
            #    cout_idx = np.expand_dims(np.arange(Cout) % 3 == i_out, 1)
            #    cin_idx = np.expand_dims(np.arange(Cin) % 3 == i_in, 0)
            #    a1, a2 = np.broadcast_arrays(cout_idx, cin_idx)
            #    return a1 * a2

            #for j in range(3):
            #    pre_mask[bmask(j, j), yc, xc] = 0.0 if mask == 'A' else 1.0

            #pre_mask[bmask(0, 1), yc, xc] = 0.0
            #pre_mask[bmask(0, 2), yc, xc] = 0.0
            #pre_mask[bmask(1, 2), yc, xc] = 0.0
        elif mask_mode == "fig1-van-den-oord":
            if vertical:
                pre_mask[:, :, yc:, :] = 0.0
            else:
                # All rows after center must be zero
                pre_mask[:, :, yc+1:, :] = 0.0
                ### All rows before center must be zero # XXX: not actually necessary
                ##pre_mask[:, :, :yc, :] = 0.0
                # All columns after center in center row must be zero
                pre_mask[:, :, yc, xc+1:] = 0.0

            if mask == 'A':
                # Center must be zero in first layer
                pre_mask[:, :, yc, xc] = 0.0
        elif mask_mode == "turukin":
            pre_mask[:, :, yc+1:, :] = 0.0
            pre_mask[:, :, yc, xc+1:] = 0.0
            if mask == 'A':
                pre_mask[:, :, yc, xc] = 0.0

        print("%s %s MASKED CONV: %d x %d. Mask:" % (mask, "VERTICAL" if vertical else "HORIZONTAL", kh, kw))
        print(pre_mask[0, 0, :, :])

        self.register_buffer("mask", torch.from_numpy(pre_mask))

    def __call__(self, x):
        self.weight.data = self.weight.data * self.mask
        return super(MaskedConvolution2D, self).forward(x)

