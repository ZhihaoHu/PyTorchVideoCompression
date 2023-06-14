from .basics import *
# import pickle
# import os
# import codecs

class Bitparm(nn.Module):
    '''
    save params
    '''
    def __init__(self, channel, final=False):
        super(Bitparm, self).__init__()
        self.final = final
        self.h = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        self.b = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        if not final:
            self.a = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        else:
            self.a = None

    def forward(self, x):
        if self.final:
            return torch.sigmoid(x * F.softplus(self.h) + self.b)
        else:
            x = x * F.softplus(self.h) + self.b
            return x + torch.tanh(x) * torch.tanh(self.a)

class ICLR17EntropyCoder(nn.Module):
    '''
    Estimate bit used in ICLR17, directly predict prob
    '''
    def __init__(self, channel):
        super(ICLR17EntropyCoder, self).__init__()
        self.f1 = Bitparm(channel)
        self.f2 = Bitparm(channel)
        self.f3 = Bitparm(channel)
        self.f4 = Bitparm(channel, True)

    def calprob(self, x):
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        return x

    def forward(self, x):
        x = Q(x, self.training)
        prob = self.calprob(x + 0.5) - self.calprob(x - 0.5)
        if self.training:
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))
        else:
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob) / math.log(2.0), 0, 50))
        return x, total_bits

class ICLR18EntropyCoder(nn.Module):
    '''
    Estimate bit used in ICLR18, use sigma to predict prob
    '''
    def __init__(self):
        super(ICLR18EntropyCoder, self).__init__()

    def forward(self, x, sigma):
        x = Q(x, self.training)
        mu = torch.zeros_like(sigma)
        sigma = sigma.clamp(1e-5, 1e10)
        gaussian = torch.distributions.laplace.Laplace(mu, sigma)
        probs = gaussian.cdf(x + 0.5) - gaussian.cdf(x - 0.5)
        if self.training:
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-5) / math.log(2.0), 0, 50))
        else:
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs) / math.log(2.0), 0, 50))
        return x, total_bits

class Parameter_net(nn.Module):
    def __init__(self):
        super(Parameter_net, self).__init__()
        self.conv1 = nn.Conv2d(256, 512, 1, stride=1, padding=0)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.conv2 = nn.Conv2d(512, 512, 1, stride=1, padding=0)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.res0 = Resblocks(512, 1)
        self.conv3 = nn.Conv2d(512, 256, 1, stride=1, padding=0)


    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.res0(x)
        return self.conv3(x)


class NIPS18nocEntropyCoder(nn.Module):
    '''
    Estimate bit used in NIPS18 without context, use mean and sigma to predict prob
    '''
    def __init__(self):
        super(NIPS18nocEntropyCoder, self).__init__()
        self.parameters_net = Parameter_net()

    def forward(self, x, musigma):
        x = Q(x, self.training)
        n,c,h,w = x.shape
        musigma = self.parameters_net(musigma)
        mu = musigma[:, 0:c, :, :]
        sigma = musigma[:, c:, :, :]
        sigma = sigma.pow(2)
        sigma = sigma.clamp(1e-5, 1e10)
        gaussian = torch.distributions.normal.Normal(mu, sigma)
        probs = gaussian.cdf(x + 0.5) - gaussian.cdf(x - 0.5)
        if self.training:
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-5) / math.log(2.0), 0, 50))
        else:
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs) / math.log(2.0), 0, 50))
        return x, total_bits


class Context_prediction_net(nn.Module):
    '''
    Compress residual prior
    '''
    def __init__(self):
        super(Context_prediction_net, self).__init__()
        self.conv1 = MaskedConvolution2D(out_channel_resM, 256, 5, stride=1, padding=2)


    def forward(self, x):
        x = self.conv1(x)
        return x

class Entropy_parameter_net(nn.Module):
    '''
    Compress residual prior
    '''
    def __init__(self):
        super(Entropy_parameter_net, self).__init__()
        self.conv1 = nn.Conv2d(512, 384, 1, stride=1, padding=0)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.conv2 = nn.Conv2d(384, 384, 1, stride=1, padding=0)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.conv3 = nn.Conv2d(384, 256, 1, stride=1, padding=0)


    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        return self.conv3(x)

class NIPS18EntropyCoder(nn.Module):
    '''
    Estimate bit used in NIPS18 without context, use mean and sigma to predict prob 
    '''
    def __init__(self):
        super(NIPS18EntropyCoder, self).__init__()
        self.context_model = Context_prediction_net()
        self.entropy_model = Entropy_parameter_net()

    def forward(self, x, musigma):
        x = Q(x, self.training)
        n,c,h,w = x.shape
        musigma = self.entropy_model(torch.cat((self.context_model(x), musigma), 1))
        mu = musigma[:, 0:c, :, :]
        sigma = musigma[:, c:, :, :]
        sigma = sigma.pow(2)
        sigma = sigma.clamp(1e-5, 1e10)
        gaussian = torch.distributions.normal.Normal(mu, sigma)
        probs = gaussian.cdf(x + 0.5) - gaussian.cdf(x - 0.5)
        if self.training:
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-5) / math.log(2.0), 0, 50))
        else:
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs) / math.log(2.0), 0, 50))
        return x, total_bits



class NIPS18nocEntropyCoder_adaptive(nn.Module):
    '''
    Estimate bit used in NIPS18 without context, use mean and sigma to predict prob
    '''
    def __init__(self):
        super(NIPS18nocEntropyCoder_adaptive, self).__init__()
        self.parameters_net = Parameter_net()
        self.maskprediction = MaskPredictionNet()

    def Q(self, x):
        if self.training:
            return x + torch.nn.init.uniform_(torch.zeros_like(x), -0.5, 0.5)
        else:
            return torch.round(x)

    def getbits(self, x, musigma, mask):
        c = x.shape[1]
        mu = musigma[:, 0:c, :, :]
        sigma = musigma[:, c:, :, :]
        sigma = sigma.pow(2)
        sigma = sigma.clamp(1e-5, 1e10)
        gaussian = torch.distributions.normal.Normal(mu, sigma)
        probs = gaussian.cdf(x + 0.5) - gaussian.cdf(x - 0.5)
        if self.training:
            bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-5) / math.log(2.0), 0, 50) * mask)
        else:
            bits = torch.sum(torch.clamp(-1.0 * torch.log(probs) / math.log(2.0), 0, 50) * mask)

        return bits

    def forward(self, x, musigma):
        n,c,h,w = x.shape
        musigma = self.parameters_net(musigma)

        mask11, mask12, mask21, mask22, mask24, mask42, mask44 = self.maskprediction(musigma)

        q_x11 = self.Q(x)
        q_x12 = self.Q(F.avg_pool2d(x, (1,2)))
        q_x21 = self.Q(F.avg_pool2d(x, (2,1)))
        q_x22 = self.Q(F.avg_pool2d(x, (2,2)))
        q_x24 = self.Q(F.avg_pool2d(x, (2,4)))
        q_x42 = self.Q(F.avg_pool2d(x, (4,2)))
        q_x44 = self.Q(F.avg_pool2d(x, (4,4)))

        shape11, shape12, shape21, shape22, shape24, shape42, shape44 = q_x11.shape[2:4], q_x12.shape[2:4], q_x21.shape[2:4], q_x22.shape[2:4], q_x24.shape[2:4], q_x42.shape[2:4], q_x44.shape[2:4]

        mask11, mask12, mask21, mask22, mask24, mask42, mask44 = F.interpolate(mask11, shape11), F.interpolate(mask12, shape12), F.interpolate(mask21, shape21), F.interpolate(mask22, shape22), F.interpolate(mask24, shape24), F.interpolate(mask42, shape42), F.interpolate(mask44, shape44)

        musigma11 = musigma
        musigma12 = F.avg_pool2d(musigma, (1,2))
        musigma21 = F.avg_pool2d(musigma, (2,1))
        musigma22 = F.avg_pool2d(musigma, (2,2))
        musigma24 = F.avg_pool2d(musigma, (2,4))
        musigma42 = F.avg_pool2d(musigma, (4,2))
        musigma44 = F.avg_pool2d(musigma, (4,4))

        num = [torch.mean(mask11), torch.mean(mask12), torch.mean(mask21), torch.mean(mask22), torch.mean(mask24), torch.mean(mask42), torch.mean(mask44)]

        bits11 = self.getbits(q_x11, musigma11, mask11)
        bits12 = self.getbits(q_x12, musigma12, mask12)
        bits21 = self.getbits(q_x21, musigma21, mask21)
        bits22 = self.getbits(q_x22, musigma22, mask22)
        bits24 = self.getbits(q_x24, musigma24, mask24)
        bits42 = self.getbits(q_x42, musigma42, mask42)
        bits44 = self.getbits(q_x44, musigma44, mask44)

        q_block_x = F.interpolate(q_x11 * mask11, shape11) + F.interpolate(q_x12 * mask12, shape11) + F.interpolate(q_x21 * mask21, shape11) + F.interpolate(q_x22 * mask22, shape11) + F.interpolate(q_x24 * mask24, shape11) + F.interpolate(q_x42 * mask42, shape11) + F.interpolate(q_x44 * mask44, shape11)

        total_bits = bits11 + bits12 + bits21 + bits22 + bits24 + bits42 + bits44

        return q_block_x, total_bits, num


class MaskPredictionNet(nn.Module):
    def __init__(self):
        super(MaskPredictionNet, self).__init__()
        self.conv0 = nn.Conv2d(out_channel_mv * 2, out_channel_mv, 3, 1, 1)
        self.res1 = Resblocks(out_channel_mv)
        self.conv1 = nn.Conv2d(out_channel_mv, out_channel_mv, 5, 2, 2)
        self.conv1_2 = nn.Conv2d(out_channel_mv, out_channel_mv * 4, 3, 1, 1)
        self.res2 = Resblocks(out_channel_mv)
        self.conv2 = nn.Conv2d(out_channel_mv, out_channel_mv, 5, 2, 2)
        self.conv2_2 = nn.Conv2d(out_channel_mv, out_channel_mv * 4, 3, 1, 1)
        self.tau = 1.

    def gumbel_softmax(self, x, hard=True, dim=1):
        if self.training:
            return F.gumbel_softmax(x, tau=self.tau, hard=hard, dim=dim)
        else:
            index = x.max(dim, keepdim=True)[1]
            return torch.zeros_like(x).scatter_(dim, index, 1.0)

    def forward(self, x):
        x = x.detach()
        n,c,h,w = x.shape

        x = self.conv0(x)

        x = self.conv1(self.res1(x))
        mode2x2 = self.conv1_2(x)
        mode2x2 = mode2x2.view(n*c//2, 4, h//2, w//2)
        mode2x2 = self.gumbel_softmax(mode2x2, hard=True, dim=1).view(n, c//2, 4, h//2, w//2)

        x = self.conv2(self.res2(x))
        mode4x4 = self.conv2_2(x)
        mode4x4 = mode4x4.view(n*c//2, 4, h//4, w//4)
        mode4x4 = self.gumbel_softmax(mode4x4, hard=True, dim=1).view(n, c//2, 4,  h//4, w//4)

        mode2x2 = mode2x2 * F.interpolate(mode4x4[:, :, 0:1, :, :], (1, h//2, w//2))

        mask11 = mode2x2[:, :, 0, :, :].contiguous()
        mask12 = mode2x2[:, :, 1, :, :].contiguous()
        mask21 = mode2x2[:, :, 2, :, :].contiguous()
        mask22 = mode2x2[:, :, 3, :, :].contiguous()

        mask24 = mode4x4[:, :, 1, :, :].contiguous()
        mask42 = mode4x4[:, :, 2, :, :].contiguous()
        mask44 = mode4x4[:, :, 3, :, :].contiguous()


        # torch.set_printoptions(threshold=np.inf)
        # gg = torch.zeros_like(mask11)
        # for mask in [mask44, mask42, mask24, mask22, mask21, mask12, mask11]:
        #     gg += F.interpolate(mask, mask11.shape[2:4])
        # print(gg)
        # print("max : ", torch.max(gg))
        # print("min : ", torch.min(gg))
        # exit(0)


        return mask11, mask12, mask21, mask22, mask24, mask42, mask44

class NIPS18nocEntropyCoder_ignore(nn.Module):
    '''
    Estimate bit used in NIPS18 without context, use mean and sigma to predict prob
    '''
    def __init__(self):
        super(NIPS18nocEntropyCoder_ignore, self).__init__()
        self.parameters_net = Parameter_net()
        self.ignore_net = IgnorePrediction()

    def forward(self, x, musigma):
        channel = x.shape[1]
        musigma = self.parameters_net(musigma)#[:, :, :h, :w].con
        ignore = self.ignore_net(musigma)
        mu = musigma[:, 0:channel, :, :]
        sigma = musigma[:, channel:, :, :]
        sigma = sigma.pow(2)
        sigma = sigma.clamp(1e-5, 1e10)
        gaussian = torch.distributions.normal.Normal(mu, sigma)
        ignored_x = x * ignore
        x = Q(x, self.training)
        probs = gaussian.cdf(ignored_x + 0.5) - gaussian.cdf(ignored_x - 0.5)
        if self.training:
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-5) / math.log(2.0), 0, 50) * (ignore.detach()))
        else:
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs) / math.log(2.0), 0, 50) * (ignore.detach()))
        return total_bits, ignore, ignored_x

class IgnorePrediction(nn.Module):
    def __init__(self):
        super(IgnorePrediction, self).__init__()
        self.conv0 = nn.Conv2d(out_channel_mv * 2, out_channel_mv, 3, 1, 1)
        self.res1 = Resblocks(out_channel_mv)
        self.conv1 = nn.Conv2d(out_channel_mv, out_channel_mv * 2, 3, 1, 1)
        self.tau = 1.

    def gumbel_softmax(self, x, hard=True, dim=1):
        if self.training:
            return F.gumbel_softmax(x, tau=self.tau, hard=hard, dim=dim)
        else:
            index = x.max(dim, keepdim=True)[1]
            return torch.zeros_like(x).scatter_(dim, index, 1.0)

    def forward(self, x):
        x = x.detach()
        n,c,h,w = x.shape
        x = self.conv0(x)
        x = self.conv1(self.res1(x))
        x = x.view(n*c//2, 2, h, w)
        x = self.gumbel_softmax(x, hard=True, dim=1)
        x = x[:, 0, :, :].view(n, c//2, h, w).contiguous()
        return x
