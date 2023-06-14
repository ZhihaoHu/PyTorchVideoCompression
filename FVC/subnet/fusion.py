
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F


def generate(c, p):
    a = torch.zeros([c*p*p, 1, p, p])
    for k in range(c):
        for i in range(p):
            for j in range(p):
                a[k * p * p + i*p + j, 0, i, j] = 1
    return a

class Nonlocal_Fusion(nn.Module):
    ''' Temporal Nonlocal Attention fusion module
    Temporal: correlation;
    '''

    def __init__(self, nf=64, nframes=4, center=0):
        super(Nonlocal_Fusion, self).__init__()
        self.center = center
        # temporal attention (before fusion conv)
        self.tAtt_1 = nn.Conv2d(nf, nf//4, 3, 1, 1, bias=True)
        self.tAtt_2 = nn.Conv2d(nf, nf//4, 3, 1, 1, bias=True)
        self.tAtt_3 = nn.Conv2d(nf, nf//4, 3, 1, 1, bias=True)
        self.patch = 3

        self.nonlocal_kernel = generate(nf//4, self.patch).cuda()

        # fusion conv: using 1x1 to save parameters and computation
        self.fea_fusion = nn.Conv2d(nframes * nf//4, nf, 1, 1, bias=True)
        self.fea_fusion2 = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)

        # spatial attention (after fusion conv)
        # self.sAtt_1 = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)#sadf
        # self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        # self.avgpool = nn.AvgPool2d(3, stride=2, padding=1)
        # self.sAtt_2 = nn.Conv2d(nf * 2, nf, 1, 1, bias=True)
        # self.sAtt_3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # self.sAtt_4 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        # self.sAtt_5 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # self.sAtt_L1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        # self.sAtt_L2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        # self.sAtt_L3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # self.sAtt_add_1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        # self.sAtt_add_2 = nn.Conv2d(nf, nf, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, aligned_fea):

        p = self.patch
        B, N, C, H, W = aligned_fea.size()  # N video frames
        C2 = C // 4
        #### temporal attention
        center_fea = aligned_fea[:, self.center, :, :, :].clone()
        return center_fea + self.fea_fusion2(aligned_fea.view(B, N*C, H, W))
        emb_ref = self.tAtt_2(center_fea).view(B, C2, 1, H ,W)
        emb = self.tAtt_1(aligned_fea.view(-1, C, H, W)).view(B, N, -1, H, W)  # [B, N, C2(nf), H, W]
        aligned_fea = self.tAtt_3(aligned_fea.view(-1, C, H, W)).view(B, N, C2, H, W)

        cor_l = []
        for i in range(N):
            emb_nbrp = emb[:, i, :, :, :]
            # cor_tmp = torch.sum(emb_nbr * emb_ref, 1).unsqueeze(1)  # B, 1, H, W
            emb_nbr = F.conv2d(emb_nbrp, self.nonlocal_kernel, padding=p//2, groups=C2).view(B, C2, p*p, H, W)
            # print(self.nonlocal_kernel)
            # for x in range(p):
            #     for y in range(self.patch):
            #         for k in range(C):
            #             print(emb_nbrp[1, k, 64+x-1, 64+y-1], emb_nbr[1, k, x*p+y, 64, 64])
            # exit()
            cor_tmp = torch.sum(emb_ref * emb_nbr, 1)  # B, p*p, H, W
            cor_l.append(cor_tmp)
        cor_prob = torch.softmax(torch.cat(cor_l, dim=1), dim=1)  # B, N*p*p, H, W
        cor_prob = cor_prob.view(B, N, -1, H, W).unsqueeze(2).repeat(1, 1, C2, 1, 1, 1).view(B, -1, H, W)# B, N*C*p*p, H, W


        aligned_fea = F.conv2d(aligned_fea.view(B * N, C2, H, W), self.nonlocal_kernel, padding=p//2, groups=C2) # B*N, C*p*p, H, W
        aligned_fea = aligned_fea.view(B, -1, H, W) * cor_prob
        aligned_fea = torch.sum(aligned_fea.view(B, N, C2, p*p, H, W), 3).view(B, -1, H, W)

        #### fusion
        fea = center_fea + self.lrelu(self.fea_fusion(aligned_fea))

        # #### spatial attention
        # att = self.lrelu(self.sAtt_1(aligned_fea))
        # att_max = self.maxpool(att)
        # att_avg = self.avgpool(att)
        # att = self.lrelu(self.sAtt_2(torch.cat([att_max, att_avg], dim=1)))
        # # pyramid levels
        # att_L = self.lrelu(self.sAtt_L1(att))
        # att_max = self.maxpool(att_L)
        # att_avg = self.avgpool(att_L)
        # att_L = self.lrelu(self.sAtt_L2(torch.cat([att_max, att_avg], dim=1)))
        # att_L = self.lrelu(self.sAtt_L3(att_L))
        # att_L = F.interpolate(att_L, scale_factor=2, mode='bilinear', align_corners=False)

        # att = self.lrelu(self.sAtt_3(att))
        # att = att + att_L
        # att = self.lrelu(self.sAtt_4(att))
        # att = F.interpolate(att, scale_factor=2, mode='bilinear', align_corners=False)
        # att = self.sAtt_5(att)
        # att_add = self.sAtt_add_2(self.lrelu(self.sAtt_add_1(att)))
        # att = torch.sigmoid(att)

        # fea = fea * att * 2 + att_add
        return fea
