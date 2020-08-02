
import os
import argparse
import torch
import cv2
import logging
import numpy as np
from net import *
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
import sys
import math
import json
from dataset import DataSet, HEVCDataSet, UVGDataSet, VTLDataSet, MCLDataSet
from tensorboardX import SummaryWriter
from drawuvg import uvgdrawplt
torch.backends.cudnn.enabled = True
# gpu_num = 4
gpu_num = torch.cuda.device_count()
cur_lr = base_lr = 1e-4#  * gpu_num
train_lambda = 2048
print_step = 100
cal_step = 10
# print_step = 10
warmup_step = 0#  // gpu_num
gpu_per_batch = 4
test_step = 10000#  // gpu_num
tot_epoch = 1000000
tot_step = 2000000
decay_interval = 1800000
lr_decay = 0.1
logger = logging.getLogger("VideoCompression")
tb_logger = None
global_step = 0
ref_i_dir = geti(train_lambda)

parser = argparse.ArgumentParser(description='DVC reimplement')

parser.add_argument('-l', '--log', default='',
        help='output training details')
parser.add_argument('-p', '--pretrain', default = '',
        help='load pretrain model')
parser.add_argument('--test', action='store_true')
parser.add_argument('--testuvg', action='store_true')
parser.add_argument('--testvtl', action='store_true')
parser.add_argument('--testmcl', action='store_true')
parser.add_argument('--testauc', action='store_true')
parser.add_argument('--rerank', action='store_true')
parser.add_argument('--allpick', action='store_true')
parser.add_argument('--config', dest='config', required=True,
        help = 'hyperparameter of Reid in json format')

def parse_config(config):
    config = json.load(open(args.config))
    global tot_epoch, tot_step, test_step, base_lr, cur_lr, lr_decay, decay_interval, train_lambda, ref_i_dir
    if 'tot_epoch' in config:
        tot_epoch = config['tot_epoch']
    if 'tot_step' in config:
        tot_step = config['tot_step']
    if 'test_step' in config:
        test_step = config['test_step']
        print('teststep : ', test_step)
    if 'train_lambda' in config:
        train_lambda = config['train_lambda']
        ref_i_dir = geti(train_lambda)
    if 'lr' in config:
        if 'base' in config['lr']:
            base_lr = config['lr']['base']
            cur_lr = base_lr
        if 'decay' in config['lr']:
            lr_decay = config['lr']['decay']
        if 'decay_interval' in config['lr']:
            decay_interval = config['lr']['decay_interval']

def adjust_learning_rate(optimizer, global_step):
    global cur_lr
    global warmup_step
    if global_step < warmup_step:
        lr = base_lr * global_step / warmup_step
    elif global_step < decay_interval:#  // gpu_num:
        lr = base_lr
    else:
        lr = base_lr * (lr_decay ** (global_step // decay_interval))
    cur_lr = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def Var(x):
    return Variable(x.cuda())

def testuvg(global_step, testfull=False):
    with torch.no_grad():
        test_loader = DataLoader(dataset=test_dataset, shuffle=False, num_workers=0, batch_size=1, pin_memory=True)
        net.eval()
        sumbpp = 0
        sumpsnr = 0
        summsssim = 0
        cnt = 0
        for batch_idx, input in enumerate(test_loader):
            if batch_idx % 10 == 0:
                print("testing : %d/%d"% (batch_idx, len(test_loader)))
            input_images = input[0]
            ref_image = input[1]
            ref_bpp = input[2]
            ref_psnr = input[3]
            ref_msssim = input[4]
            seqlen = input_images.size()[1]
            sumbpp += torch.mean(ref_bpp).detach().numpy()
            sumpsnr += torch.mean(ref_psnr).detach().numpy()
            summsssim += torch.mean(ref_msssim).detach().numpy()
            cnt += 1
            for i in range(seqlen):
                input_image = input_images[:, i, :, :, :]
                inputframe, refframe = Var(input_image), Var(ref_image)
                clipped_recon_image, mse_loss, warploss, interloss, bpp_feature, bpp_z, bpp_mv, bpp = net(inputframe, refframe)
                sumbpp += torch.mean(bpp).cpu().detach().numpy()
                sumpsnr += torch.mean(10 * (torch.log(1. / mse_loss) / np.log(10))).cpu().detach().numpy()
                summsssim += ms_ssim(clipped_recon_image.cpu().detach(), input_image, data_range=1.0, size_average=True).numpy()
                cnt += 1
                ref_image = clipped_recon_image
        log = "global step %d : " % (global_step) + "\n"
        logger.info(log)
        sumbpp /= cnt
        sumpsnr /= cnt
        summsssim /= cnt
        log = "UVGdataset : average bpp : %.6lf, average psnr : %.6lf, average msssim: %.6lf\n" % (sumbpp, sumpsnr, summsssim)
        logger.info(log)
        uvgdrawplt([sumbpp], [sumpsnr], [summsssim], global_step, testfull=testfull)


def train(epoch, global_step):

    print ("epoch", epoch)
    global gpu_per_batch
    train_loader = DataLoader(dataset = train_dataset, shuffle=True, num_workers=gpu_num, batch_size=gpu_per_batch, pin_memory=True)
    net.train()

    global optimizer
    bat_cnt = 0
    cal_cnt = 0
    sumloss = 0
    sumpsnr = 0
    suminterpsnr = 0
    sumwarppsnr = 0
    sumbpp = 0
    sumbpp_feature = 0
    sumbpp_mv = 0
    sumbpp_z = 0
    tot_iter = len(train_loader)
    t0 = datetime.datetime.now()
    for batch_idx, input in enumerate(train_loader):
        global_step += 1
        bat_cnt += 1
        input_image, ref_image = Var(input[0]), Var(input[1])
        quant_noise_feature, quant_noise_z, quant_noise_mv = Var(input[2]), Var(input[3]), Var(input[4])
        # ta = datetime.datetime.now()
        clipped_recon_image, mse_loss, warploss, interloss, bpp_feature, bpp_z, bpp_mv, bpp = net(input_image, ref_image, quant_noise_feature, quant_noise_z, quant_noise_mv)
        
        # tb = datetime.datetime.now()
        mse_loss, warploss, interloss, bpp_feature, bpp_z, bpp_mv, bpp = \
            torch.mean(mse_loss), torch.mean(warploss), torch.mean(interloss), torch.mean(bpp_feature), torch.mean(bpp_z), torch.mean(bpp_mv), torch.mean(bpp)
        distribution_loss = bpp
        if global_step < 500000:
            warp_weight = 0.1
        else:
            warp_weight = 0
        distortion = mse_loss + warp_weight * (warploss + interloss)
        rd_loss = train_lambda * distortion + distribution_loss
        # tc = datetime.datetime.now()
        optimizer.zero_grad()
        rd_loss.backward()
        # tf = datetime.datetime.now()
        def clip_gradient(optimizer, grad_clip):
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        param.grad.data.clamp_(-grad_clip, grad_clip)
        clip_gradient(optimizer, 0.5)
        optimizer.step()
        if global_step % cal_step == 0:
            cal_cnt += 1
            if mse_loss > 0:
                psnr = 10 * (torch.log(1 * 1 / mse_loss) / np.log(10)).cpu().detach().numpy()
            else:
                psnr = 100
            if warploss > 0:
                warppsnr = 10 * (torch.log(1 * 1 / warploss) / np.log(10)).cpu().detach().numpy()
            else:
                warppsnr = 100
            if interloss > 0:
                interpsnr = 10 * (torch.log(1 * 1 / interloss) / np.log(10)).cpu().detach().numpy()
            else:
                interpsnr = 100

            loss_ = rd_loss.cpu().detach().numpy()

            sumloss += loss_
            sumpsnr += psnr
            suminterpsnr += interpsnr
            sumwarppsnr += warppsnr
            sumbpp += bpp.cpu().detach()
            sumbpp_feature += bpp_feature.cpu().detach()
            sumbpp_mv += bpp_mv.cpu().detach()
            sumbpp_z += bpp_z.cpu().detach()


        if (batch_idx % print_step)== 0 and bat_cnt > 1:
            tb_logger.add_scalar('lr', cur_lr, global_step)
            tb_logger.add_scalar('rd_loss', sumloss / cal_cnt, global_step)
            tb_logger.add_scalar('psnr', sumpsnr / cal_cnt, global_step)
            tb_logger.add_scalar('warppsnr', sumwarppsnr / cal_cnt, global_step)
            tb_logger.add_scalar('interpsnr', suminterpsnr / cal_cnt, global_step)
            tb_logger.add_scalar('bpp', sumbpp / cal_cnt, global_step)
            tb_logger.add_scalar('bpp_feature', sumbpp_feature / cal_cnt, global_step)
            tb_logger.add_scalar('bpp_z', sumbpp_z / cal_cnt, global_step)
            tb_logger.add_scalar('bpp_mv', sumbpp_mv / cal_cnt, global_step)
            t1 = datetime.datetime.now()
            deltatime = t1 - t0
            log = 'Train Epoch : {:02} [{:4}/{:4} ({:3.0f}%)] Avgloss:{:.6f} lr:{} time:{}'.format(epoch, batch_idx, len(train_loader), 100. * batch_idx / len(train_loader), sumloss / cal_cnt, cur_lr, (deltatime.seconds + 1e-6 * deltatime.microseconds) / bat_cnt)
            print(log)
            log = 'details : warppsnr : {:.2f} interpsnr : {:.2f} psnr : {:.2f}'.format(sumwarppsnr / cal_cnt, suminterpsnr / cal_cnt, sumpsnr / cal_cnt)
            print(log)
            bat_cnt = 0
            cal_cnt = 0
            sumbpp = sumbpp_feature = sumbpp_mv = sumbpp_z = sumloss = sumpsnr = suminterpsnr = sumwarppsnr = 0
            t0 = t1
    log = 'Train Epoch : {:02} Loss:\t {:.6f}\t lr:{}'.format(epoch, sumloss / bat_cnt, cur_lr)
    logger.info(log)
    return global_step


if __name__ == "__main__":
    args = parser.parse_args()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s] %(message)s')
    stdhandler = logging.StreamHandler()
    stdhandler.setLevel(logging.INFO)
    stdhandler.setFormatter(formatter)
    logger.addHandler(stdhandler)
    if args.log != '':
        filehandler = logging.FileHandler(args.log)
        filehandler.setLevel(logging.INFO)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)
    logger.setLevel(logging.INFO)
    logger.info("DVC training")
    logger.info("config : ")
    logger.info(open(args.config).read())
    parse_config(args.config)

    model = VideoCompressor()
    if args.pretrain != '':
        print("loading pretrain : ", args.pretrain)
        global_step = load_model(model, args.pretrain)
    net = model.cuda()
    net = torch.nn.DataParallel(net, list(range(gpu_num)))
    bp_parameters = net.parameters()
    optimizer = optim.Adam(bp_parameters, lr=base_lr)
    # save_model(model, 0)
    global train_dataset, test_dataset
    if args.testuvg:
        test_dataset = UVGDataSet(refdir=ref_i_dir, testfull=True)
        print('testing UVG')
        testuvg(0, testfull=True)
        exit(0)

    tb_logger = SummaryWriter('./events')
    train_dataset = DataSet("data/vimeo_septuplet/test.txt")
    # test_dataset = UVGDataSet(refdir=ref_i_dir)
    stepoch = global_step // (train_dataset.__len__() // (gpu_per_batch))# * gpu_num))
    for epoch in range(stepoch, tot_epoch):
        adjust_learning_rate(optimizer, global_step)
        if global_step > tot_step:
            save_model(model, global_step)
            break
        global_step = train(epoch, global_step)
        save_model(model, global_step)
