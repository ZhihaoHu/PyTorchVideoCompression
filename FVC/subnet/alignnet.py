import torch
import torch.nn as nn
import torch.nn.functional as F
from .basics import *
from .offsetcoder import OffsetEncodeNet, OffsetDecodeNet

try:
    from .dcn.deform_conv import ModulatedDeformConvPack as DCN
    from .dcn.deform_conv import DeformConvPack as DCNv1
except ImportError:
    raise ImportError('Failed to import DCN module.')


class FeatureEncoder(nn.Module):
    '''
    Feature Encoder
    '''
    def __init__(self, nf=out_channel_M):
        super(FeatureEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, nf, 5, 2, 2)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)
        self.feature_extraction = Resblocks(nf)

    def forward(self, x):
        x = self.conv1(x)
        return self.feature_extraction(x)



class FeatureDecoder(nn.Module):
    '''
    Feature Decoder
    '''
    def __init__(self, nf=out_channel_M):
        super(FeatureDecoder, self).__init__()
        self.recon_trunk = Resblocks(nf)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)
        self.deconv1 = nn.ConvTranspose2d(nf, 3, 5, stride=2, padding=2, output_padding=1)

    def forward(self, x):
        x = self.recon_trunk(x)
        x = self.deconv1(x)
        return x


class PCD_Align(nn.Module):
    '''
    Alignment module using Pyramid, Cascading and Deformable convolution
    with 3 pyramid levels.
    '''

    def __init__(self, nf=out_channel_M, groups=8, compressoffset=True):
        super(PCD_Align, self).__init__()
        self.compressoffset = compressoffset

        self.offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.offset_encoder = OffsetEncodeNet()
        self.offset_decoder = OffsetDecodeNet()
        self.deformable_convolution = DCNv1(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                              extra_offset_mask=True)
        self.refine_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1)  # concat for diff
        self.refine_conv2 = nn.Conv2d(nf, nf, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1)


    def MotionEstimation(self, ref_fea, inp_fea):
        # motion estimation
        input_offset = torch.cat([ref_fea, inp_fea], dim=1)
        input_offset = self.lrelu(self.offset_conv1(input_offset))
        input_offset = self.lrelu(self.offset_conv3(input_offset))

        # motion compression
        en_offset = self.offset_encoder(input_offset)
        return en_offset
    
    def MotionCompensation(self, ref_fea, q_offset):
        de_offset = self.offset_decoder(q_offset)

        # motion compensation
        aligned_feature = self.deformable_convolution([ref_fea, de_offset])

        refine_feature = torch.cat([aligned_feature, ref_fea], dim=1)
        refine_feature = self.lrelu(self.refine_conv1(refine_feature))
        refine_feature = self.lrelu(self.refine_conv2(refine_feature))
        aligned_feature = aligned_feature + refine_feature

        return aligned_feature

    def forward(self, ref_fea, inp_fea):
        # motion estimation
        input_offset = torch.cat([ref_fea, inp_fea], dim=1)
        input_offset = self.lrelu(self.offset_conv1(input_offset))
        input_offset = self.lrelu(self.offset_conv3(input_offset))

        # motion compression
        en_offset = self.offset_encoder(input_offset)
        q_offset = self.Q(en_offset)
        de_offset = self.offset_decoder(q_offset)

        # motion compensation
        aligned_feature = self.deformable_convolution([ref_fea, de_offset])

        refine_feature = torch.cat([aligned_feature, ref_fea], dim=1)
        refine_feature = self.lrelu(self.refine_conv1(refine_feature))
        refine_feature = self.lrelu(self.refine_conv2(refine_feature))
        aligned_feature = aligned_feature + refine_feature

        return aligned_feature, en_offset, q_offset
