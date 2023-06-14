import math
import torch.nn as nn
import torch
from .basics import *


class ResEncodeNet(nn.Module):
    '''
    Residual Encode Net
    '''
    def __init__(self):
        super(ResEncodeNet, self).__init__()
        self.conv1 = nn.Conv2d(out_channel_M, out_channel_resN, 5, stride=2, padding=2)
        self.res1 = Resblocks()
        self.conv2 = nn.Conv2d(out_channel_resN, out_channel_resN, 5, stride=2, padding=2)
        self.res2 = Resblocks()
        self.conv3 = nn.Conv2d(out_channel_resN, out_channel_resN, 5, stride=2, padding=2)


    def forward(self, x):
        x = self.res1(self.conv1(x))
        x = self.res2(self.conv2(x))
        # x = Padit(x)
        return self.conv3(x)


class ResDecodeNet(nn.Module):
    '''
    Residual Decode Net
    '''
    def __init__(self):
        super(ResDecodeNet, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(out_channel_resM, out_channel_resN, 5, stride=2, padding=2, output_padding=1)
        self.res1 = Resblocks()
        self.deconv2 = nn.ConvTranspose2d(out_channel_resN, out_channel_resN, 5, stride=2, padding=2, output_padding=1)
        self.res2 = Resblocks()
        self.deconv3 = nn.ConvTranspose2d(out_channel_resN, out_channel_M, 5, stride=2, padding=2, output_padding=1)

    def forward(self, x):
        x = self.res1(self.deconv1(x))
        x = self.res2(self.deconv2(x))
        x = self.deconv3(x)
        return x



class ResPriorEncodeNet(nn.Module):
    '''
    Residual Prior Encode Net
    '''
    def __init__(self):
        super(ResPriorEncodeNet, self).__init__()
        self.conv1 = nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=1, padding=1)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.conv2 = nn.Conv2d(out_channel_mv, out_channel_mv, 5, stride=2, padding=2)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2)
        self.conv3 = nn.Conv2d(out_channel_mv, out_channel_mv, 5, stride=2, padding=2)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        # x = Padit(x)
        x = self.relu2(self.conv2(x))
        # x = Padit(x)
        return self.conv3(x)

class ResPriorDecodeNet(nn.Module):
    '''
    Residual Prior Decode Net
    '''
    def __init__(self):
        super(ResPriorDecodeNet, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(out_channel_mv, out_channel_mv, 5, stride=2, padding=2, output_padding=1)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.deconv2 = nn.ConvTranspose2d(out_channel_mv, out_channel_mv * 3 // 2, 5, stride=2, padding=2, output_padding=1)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2)
        self.deconv3 = nn.ConvTranspose2d(out_channel_mv * 3 // 2, out_channel_mv * 2, 3, stride=1, padding=1)

    def forward(self, x):
        x = self.relu1(self.deconv1(x))
        x = self.relu2(self.deconv2(x))
        return self.deconv3(x)
