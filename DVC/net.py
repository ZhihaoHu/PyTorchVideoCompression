import numpy as np
import os
import torch
import torchvision.models as models
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import math
import torch.nn.init as init
import logging
from torch.nn.parameter import Parameter
from subnet import *

def save_model(model, iter):
    torch.save(model.state_dict(), "./snapshot/iter{}.model".format(iter))

def load_model(model, f):
    with open(f, 'rb') as f:
        pretrained_dict = torch.load(f)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    f = str(f)
    if f.find('iter') != -1 and f.find('.model') != -1:
        st = f.find('iter') + 4
        ed = f.find('.model', st)
        return int(f[st:ed])
    else:
        return 0



class VideoCompressor(nn.Module):
    def __init__(self):
        super(VideoCompressor, self).__init__()
        # self.imageCompressor = ImageCompressor()
        self.opticFlow = ME_Spynet()
        self.mvEncoder = Analysis_mv_net()
        self.Q = None
        self.mvDecoder = Synthesis_mv_net()
        self.warpnet = Warp_net()
        self.resEncoder = Analysis_net()
        self.resDecoder = Synthesis_net()
        self.respriorEncoder = Analysis_prior_net()
        self.respriorDecoder = Synthesis_prior_net()
        self.bitEstimator_z = BitEstimator(out_channel_N)
        self.bitEstimator_mv = BitEstimator(out_channel_mv)
        # self.flow_warp = Resample2d()
        # self.bitEstimator_feature = BitEstimator(out_channel_M)
        self.warp_weight = 0

    def forwardFirstFrame(self, x):
        output, bittrans = self.imageCompressor(x)
        cost = self.bitEstimator(bittrans)
        return output, cost

    def motioncompensation(self, ref, mv):
        warpframe = flow_warp(ref, mv)
        inputfeature = torch.cat((warpframe, ref), 1)
        prediction = self.warpnet(inputfeature) + warpframe
        return prediction, warpframe

    def forward(self, input_image, referframe, quant_noise_feature=None, quant_noise_z=None, quant_noise_mv=None):
        estmv = self.opticFlow(input_image, referframe)
        mvfeature = self.mvEncoder(estmv)
        if self.training:
            quant_mv = mvfeature + quant_noise_mv
        else:
            quant_mv = torch.round(mvfeature)
        quant_mv_upsample = self.mvDecoder(quant_mv)

        prediction, warpframe = self.motioncompensation(referframe, quant_mv_upsample)

        input_residual = input_image - prediction

        feature = self.resEncoder(input_residual)
        batch_size = feature.size()[0]

        z = self.respriorEncoder(feature)

        if self.training:
            compressed_z = z + quant_noise_z
        else:
            compressed_z = torch.round(z)

        recon_sigma = self.respriorDecoder(compressed_z)

        feature_renorm = feature

        if self.training:
            compressed_feature_renorm = feature_renorm + quant_noise_feature
        else:
            compressed_feature_renorm = torch.round(feature_renorm)

        recon_res = self.resDecoder(compressed_feature_renorm)
        recon_image = prediction + recon_res

        clipped_recon_image = recon_image.clamp(0., 1.)


# distortion
        mse_loss = torch.mean((recon_image - input_image).pow(2))

        # psnr = tf.cond(
        #     tf.equal(mse_loss, 0), lambda: tf.constant(100, dtype=tf.float32),
        #     lambda: 10 * (tf.log(1 * 1 / mse_loss) / np.log(10)))

        warploss = torch.mean((warpframe - input_image).pow(2))
        interloss = torch.mean((prediction - input_image).pow(2))
        # psnr_warp = tf.cond(
        #     tf.equal(warploss, 0), lambda: tf.constant(100, dtype=tf.float32),
        #     lambda: 10 * (tf.log(1 * 1 / warploss) / np.log(10)))

        # ssim_r = tf_ms_ssim(recon_image[..., 0:1], input_image[..., 0:1])
        # ssim_g = tf_ms_ssim(recon_image[..., 1:2], input_image[..., 1:2])
        # ssim_b = tf_ms_ssim(recon_image[..., 2:3], input_image[..., 2:3])
        # ms_ssim = (ssim_r + ssim_g + ssim_b) / 3
        # ms_ssim_db = -10 * tf.log(1 - ms_ssim) / np.log(10)

        # 117w
        # 50Wfor mse+warp, then mse
        # distortion = mse_loss + 0.1 * warploss
        # distortion = mse_loss + self.warp_weight * warploss

# bit per pixel

        def feature_probs_based_sigma(feature, sigma):
            mu = torch.zeros_like(sigma)
            sigma = sigma.clamp(1e-5, 1e10)
            gaussian = torch.distributions.laplace.Laplace(mu, sigma)
            probs = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-5) / math.log(2.0), 0, 50))
            return total_bits, probs

        def iclr18_estrate_bits_z(z):
            prob = self.bitEstimator_z(z + 0.5) - self.bitEstimator_z(z - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))
            return total_bits, prob


        def iclr18_estrate_bits_mv(mv):
            prob = self.bitEstimator_mv(mv + 0.5) - self.bitEstimator_mv(mv - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))
            return total_bits, prob

        total_bits_feature, _ = feature_probs_based_sigma(compressed_feature_renorm, recon_sigma)
        # entropy_context = entropy_context_from_sigma(compressed_feature_renorm, recon_sigma)
        total_bits_z, _ = iclr18_estrate_bits_z(compressed_z)
        total_bits_mv, _ = iclr18_estrate_bits_mv(quant_mv)

        im_shape = input_image.size()

        bpp_feature = total_bits_feature / (batch_size * im_shape[2] * im_shape[3])
        bpp_z = total_bits_z / (batch_size * im_shape[2] * im_shape[3])
        bpp_mv = total_bits_mv / (batch_size * im_shape[2] * im_shape[3])
        bpp = bpp_feature + bpp_z + bpp_mv
        
        # distribution_loss = bpp
        # rd_loss = train_lambda * distortion + distribution_loss

        return clipped_recon_image, mse_loss, warploss, interloss, bpp_feature, bpp_z, bpp_mv, bpp
        # return quant_mv_upsample, clipped_recon_image, mse_loss, warploss, interloss, bpp_feature, bpp_z, bpp_mv, bpp