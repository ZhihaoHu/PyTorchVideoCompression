
import os
import argparse
import torch
import cv2
import logging
import numpy as np
from net import *
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import DataSet, UVGDataSet
from tensorboardX import SummaryWriter
from draw import hevcdrawplt
gpu_num = torch.cuda.device_count()
cur_lr = base_lr = 1e-4
train_lambda = 2048
print_step = 100
cal_step = 10
warmup_step = 0
gpu_per_batch = 4
test_step = 10000
tot_epoch = 1000000
tot_step = 2000000
single_training_step = 2000000
decay_interval = 1800000
lr_decay = 0.1
logger = logging.getLogger("VideoCompression")
tb_logger = None
global_step = 0
ref_i_dir = geti(train_lambda)
clip_grad = 0.5
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='FVC')

parser.add_argument('-l', '--log', default='',
        help='output training details')
parser.add_argument('-p', '--pretrain', default='',
        help='load pretrain model')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--test', action='store_true')
parser.add_argument('--testuvg', action='store_true')
parser.add_argument('--testvtl', action='store_true')
parser.add_argument('--testmcl', action='store_true')
parser.add_argument('--testauc', action='store_true')
parser.add_argument('--config', dest='config', required=True,
        help = 'hyperparameter of Reid in json format')
parser.add_argument('--seed', default=1234, type=int, help='seed for random functions, and network initialization')


def parse_config(config):
    config = json.load(open(config))
    global tot_epoch, tot_step, test_step, base_lr, cur_lr, lr_decay, decay_interval, train_lambda, ref_i_dir, clip_grad, single_training_step
    if 'tot_epoch' in config:
        tot_epoch = config['tot_epoch']
    if 'tot_step' in config:
        tot_step = config['tot_step']
    if 'test_step' in config:
        test_step = config['test_step']
        print('test step : ', test_step)
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
    if 'single_training_step' in config:
        single_training_step = config['single_training_step']
    if 'clip_grad' in config:
        clip_grad = config['clip_grad']



def adjust_learning_rate(optimizer, global_step):
    global cur_lr
    lr = base_lr
    for i in range(len(decay_interval)):
        if global_step < decay_interval[i]:
            break
        else:
            lr = base_lr * lr_decay[i]
    lr = round(float(lr), 8)
    cur_lr = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def Test(name, test_dataset, global_step=0, testfull=False):
    with torch.no_grad():
        test_loader = DataLoader(dataset=test_dataset, shuffle=False, num_workers=1, batch_size=1, pin_memory=True)
        net.eval()

        sumout = dict()
        cnt = 0
        for batch_idx, inputs in enumerate(test_loader):
            print("testing : %d/%d"% (batch_idx, len(test_loader)))
            input_images = inputs[0].cuda()
            ref_image = inputs[1].cuda()
            ref = inputs[2]
            for k, v in ref.items():
                if k in sumout:
                    sumout[k] += v.detach().numpy()
                else:
                    sumout[k] = v.detach().numpy()
            seqlen = input_images.shape[1]
            cnt += 1
            for i in range(seqlen):
                input_image = input_images[:, i, :, :, :]

                recon_image, out = net(ref_image, input_image)
                out["msssim"] = ms_ssim(recon_image, input_image, data_range=1.0, size_average=True)

                for k in out:
                    out[k] = out[k].cpu().detach().numpy()

                out["psnr"] = MSE2PSNR(out["mse_loss"])

                for key, value in out.items():
                    if key in sumout:
                        sumout[key] += value
                    else:
                        sumout[key] = value

                cnt += 1
                ref_image = recon_image

        if global_step > 0:
            log = "global step %d : " % (global_step) + "\n"
            logger.info(log)
        for k in sumout:
            sumout[k] /= cnt
        log = "%s : average bpp : %.6lf, average psnr : %.6lf, average msssim: %.6lf\n" % (name, sumout["bpp"], sumout["psnr"], sumout["msssim"])
        logger.info(log)

        if tb_logger != None and not testfull:
            tb_logger.add_scalar(name + " bpp", sumout["bpp"], global_step)
            tb_logger.add_scalar(name + " psnr", sumout["psnr"], global_step)
            tb_logger.add_scalar(name + " msssim", sumout["msssim"], global_step)
            hevcdrawplt(name, [sumout["bpp"]], [sumout["psnr"]], [sumout["msssim"]], tb_logger, global_step, testfull=testfull)


def train(epoch, global_step):
    print ("epoch", epoch)
    global gpu_per_batch
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, num_workers=gpu_num*4, batch_size=gpu_per_batch, pin_memory=True)
    net.train()
    global optimizer
    bat_cnt = cal_cnt = 0
    sumout = dict()

    t0 = datetime.datetime.now()
    for batch_idx, inputs in enumerate(train_loader):
        if global_step % 100 == 0:
            net.Trainstage(global_step)

        global_step += 1
        bat_cnt += 1
        ref_gt, input_images = Var(inputs[0]), Var(inputs[1])

        input_images = list(input_images.split(3, dim=1))
        numimg = len(input_images)
        seqout = dict()

        ref_image = ref_gt


        Tweight = 1
        total = len(input_images)
        for input_image in input_images:
            Tweight += 1
            net.true_lambda = net.train_lambda * Tweight * 2 / (total + 3)

            recon_image, out = net(ref_image, input_image)

            for key, value in out.items():
                if key in seqout:
                    seqout[key] += torch.mean(value)
                else:
                    seqout[key] = torch.mean(value)

            ref_image = recon_image

        for key in out:
            seqout[key] /= numimg

        optimizer.zero_grad()
        seqout["rd_loss"].backward()

        clip_gradient(optimizer, clip_grad)
        optimizer.step()
        if global_step % cal_step == 0:
            cal_cnt += 1
            for key in seqout:
                seqout[key] = seqout[key].cpu().detach().numpy()

            seqout["psnr"] = MSE2PSNR(seqout["mse_loss"])
            seqout["psnr_align"] = MSE2PSNR(seqout["align_loss"])
            seqout["lr"] = cur_lr

            for key, value in seqout.items():
                if key in sumout:
                    sumout[key] += value
                else:
                    sumout[key] = value

        if ((batch_idx % print_step) == 0 and cal_cnt > 0):
            adjust_learning_rate(optimizer, global_step)
            for key in sumout:
                sumout[key] /= cal_cnt

            for key, value in sumout.items():
                tb_logger.add_scalar(key, value, global_step)
            t1 = datetime.datetime.now()
            deltatime = t1 - t0
            t0 = t1
            log = 'Train Epoch : {:02} [{:4}/{:4} ({:3.0f}%)] loss:{:.6f} lr:{} time:{:.3f}'.format(epoch, batch_idx, len(train_loader), 100. * batch_idx / len(train_loader), sumout["rd_loss"], cur_lr, (deltatime.seconds + 1e-6 * deltatime.microseconds) / bat_cnt)
            print(log)
            log = 'details : psnr_a : {:.2f} psnr : {:.2f}, bpp_off : {:.2f}, bpp_res : {:.2f}'.format(sumout["psnr_align"], sumout["psnr"], sumout["bpp_offsetf"] + sumout["bpp_offsetz"], sumout["bpp_residualf"] + sumout["bpp_residualz"])
            print(log)
            bat_cnt = 0
            cal_cnt = 0
            sumout = dict()

        if global_step >= tot_step or global_step == single_training_step:
            break
    log = "Train Epoch : {:02} Step : {}".format(epoch, global_step)
    logger.info(log)
    return global_step


if __name__ == "__main__":
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

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
    logger.info("Video Compression training")
    logger.info("config : ")
    logger.info(open(args.config).read())
    parse_config(args.config)

    model = VideoCompressor(train_lambda)
    net = model.cuda()
    bp_parameters = net.parameters()
    optimizer = optim.Adam(bp_parameters, lr=base_lr)
    print("model parameter : ", sum(p.numel() for p in model.parameters()))

    if args.resume:
        global_step = resume(model, optimizer, "snapshot/latest.model")
    elif args.pretrain != '':
        print("loading pretrain : ", args.pretrain)
        global_step = load_model(model, args.pretrain)

    # net = torch.nn.DataParallel(net, list(range(gpu_num)))

    global train_dataset
    if args.testuvg:
        print('testing UVG Dataset : ')
        test_dataset = UVGDataSet(refdir=ref_i_dir, testfull=True)
        Test("UVG dataset", test_dataset, 0, testfull=True)
        exit(0)
    tb_logger = SummaryWriter('./events')
    train_dataset = DataSet()
    if global_step >= single_training_step:
        gpu_per_batch = 2
        train_dataset.multiframeTrain()
    epoch = global_step // (train_dataset.__len__() // (gpu_per_batch))
    while True:
        if global_step >= tot_step:
            save_model(model, optimizer, global_step)
            break

        adjust_learning_rate(optimizer, global_step)

        global_step = train(epoch, global_step)

        if global_step == single_training_step:
            gpu_per_batch = 2
            train_dataset.multiframeTrain()

        if epoch % 2 == 0:
            save_model(model, optimizer, global_step)

        epoch += 1

