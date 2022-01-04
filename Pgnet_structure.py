"""
@author: lsl
E-mail: cug_lsl@cug.edu.cn
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

up_ratio = 2  # 基本不变
kernelsize_temp = 3
kernelsize_temp2 = 5  # 空间注意力细节程度，越大细节越大
padding_mode = 'circular'

def log(base, x):
    return np.log(x) / np.log(base)


class simple_net(nn.Module):
    def __init__(self,
                 input_channel: int,
                 output_channel: int,
                 kernelsize: int = 3):
        super(simple_net, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernelsize, stride=1, padding_mode=padding_mode,
                      padding=int(kernelsize // 2)),
            nn.BatchNorm2d(output_channel),
            nn.LeakyReLU())

    def forward(self, x: torch.Tensor):
        return self.net(x)

class simple_net_res(nn.Module):
    def __init__(self,
                 input_channel: int,
                 output_channel: int,
                 kernelsize: int = 3):
        super(simple_net_res, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernelsize, stride=1, padding_mode=padding_mode,
                      padding=int(kernelsize // 2)),
            nn.LeakyReLU())

    def forward(self, x: torch.Tensor):
        return self.net(x)


class basic_net(nn.Module):
    def __init__(self,
                 input_channel: int,
                 output_channel: int,
                 mid_channel: int = 64,
                 kernelsize=kernelsize_temp):
        super(basic_net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, mid_channel, kernelsize, stride=1, padding_mode=padding_mode,
                      padding=int(kernelsize // 2)),
            nn.BatchNorm2d(mid_channel),
            nn.LeakyReLU())  # Lrelu
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channel, output_channel, kernelsize, stride=1, padding_mode=padding_mode,
                      padding=int(kernelsize // 2)),
            nn.BatchNorm2d(output_channel))

    def forward(self, x: torch.Tensor):
        return self.conv2(self.conv1(x))

class basic_net_nobn(nn.Module):
    def __init__(self,
                 input_channel: int,
                 output_channel: int,
                 mid_channel: int = 64,
                 kernelsize=kernelsize_temp):
        super(basic_net_nobn, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, mid_channel, kernelsize, stride=1, padding_mode=padding_mode,
                      padding=int(kernelsize // 2)),
            nn.LeakyReLU())  # Lrelu
        self.conv2 = nn.Conv2d(mid_channel, output_channel, kernelsize, stride=1, padding_mode=padding_mode,
                      padding=int(kernelsize // 2))

    def forward(self, x: torch.Tensor):
        temp = self.conv1(x)
        temp2 = self.conv2(temp)
        return temp2

# 专门处理全色波段的降采样basic net  与反卷积一起使用
class pan_net(nn.Module):
    def __init__(self,
                 input_channel: int,
                 output_channel: int,
                 mid_channel: int = 64,
                 kernelsize=3,
                 stride0=1,  # 控制着降采样ratio
                 stride1=1):  # 控制着降采样ratio
        super(pan_net, self).__init__()
        self.output_ch = output_channel

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, mid_channel, kernelsize, stride=[stride0, stride0],
                      padding_mode=padding_mode,
                      padding=int(kernelsize // 2)),
            nn.BatchNorm2d(mid_channel),
            nn.LeakyReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channel, output_channel, kernelsize, stride=[stride1, stride1],
                      padding_mode=padding_mode,
                      padding=int(kernelsize // 2)),
            nn.BatchNorm2d(output_channel))

    def forward(self, x: torch.Tensor):
        return self.conv2(self.conv1(x))


class res_dense_net(nn.Module):
    def __init__(self,
                 in_size,
                 pan_size,
                 endmember_num=32,
                 kernelsize=3,
                 pan_dim=1,
                 ratio=4,
                 padding=2):
        super(res_dense_net, self).__init__()

        self.deconv = nn.Sequential(
            nn.Upsample(scale_factor=ratio, mode='bicubic'),
            simple_net(endmember_num, endmember_num, 1))

        self.pan_pro = pan_net(1, pan_dim, mid_channel=32, stride0=int(np.sqrt(pan_size/(ratio*in_size))),
                            stride1=int(np.sqrt(pan_size/(ratio*in_size))))

        self.fuse = pan_abunstd(endmember_num=endmember_num)

    def forward(self, w1, w2, b1, b2, abun, pan, dim_band=1):

        return self.fuse(w1, w2, b1, b2, self.pan_pro(pan), self.deconv(abun))

class pan_net2(nn.Module):  # pixel注意力  乘
    def __init__(self,
                 endmember_num=32,
                 pan_dim=16,
                 kernelsize=3
                 ):
        super(pan_net2, self).__init__()

        # self.conv1 = basic_net(1, pan_dim, mid_channel=32)  # 全色升维

        self.fuse2 = pan_abunstd(endmember_num=endmember_num)

        self.weight = simple_net(endmember_num, endmember_num)  # 得到像元权重

        self.act = nn.Sigmoid()
        # self.act = nn.Tanh()

    def forward(self, w1, w2, b1, b2, abun, pan, dim_band=1):

        # temp1 = self.fuse2(self.conv1(pan), abun)     ################## 待测试 #################
        temp1 = self.fuse2(w1, w2, b1, b2, pan, abun)
        return temp1 * self.act(self.weight(temp1)) + abun


# PDIN
class pan_abunstd(nn.Module):
    def __init__(self, endmember_num=30):
        super(pan_abunstd, self).__init__()

        self.multiply_weight = simple_net_res(1, endmember_num, kernelsize=3)

        self.plus_weight = simple_net_res(1, endmember_num, kernelsize=3)


    def forward(self, w1, w2, b1, b2, pan, abun, dim_band=1):

        # print(pan.shape)
        update_weight = F.sigmoid(F.relu(pan - w1 * torch.std(abun, dim=1, keepdim=True) - b1))
        update_weight2 = F.sigmoid(F.relu(w2 * torch.std(abun, dim=1, keepdim=True) + b2 - pan))

        update_weight = update_weight + update_weight2

        result0 = self.multiply_weight(pan) * update_weight * abun + self.plus_weight(pan) * update_weight

        return result0 + abun 


class Pg_net(nn.Module):
    def __init__(self, band=126, endmember_num=10, ratio=16, abun_block_num=4, pan_dim=1,
                hs_size=4, pan_size=64, up_chan_num=10, up_ratio=2):
        super(Pg_net, self).__init__()
        self.band = band
        self.endmember_num = endmember_num
        self.up_chan_num = up_chan_num
        self.ratio = ratio
        self.upscale_num = int(log(up_ratio, ratio) + 1)
        self.abun_block_num = abun_block_num  # attention block number
        self.pan_dim = pan_dim
        self.pan_blocks0 = 1  # every attention block could have more sub-attention block

        self.up_ratio = up_ratio

        self.w1 = nn.Parameter(torch.tensor(1.0))
        self.w2 = nn.Parameter(torch.tensor(-1.0))

        self.b1 = nn.Parameter(torch.tensor(0.3))
        self.b2 = nn.Parameter(torch.tensor(0.3))

        self.upsample16 = nn.Upsample(scale_factor=self.ratio, mode='bicubic')  # 丰度的上采样

        def pan_dict(endmember_num, pan_dim):
            return nn.ModuleList(
                [pan_net2(endmember_num=endmember_num, pan_dim=pan_dim, kernelsize=kernelsize_temp) for _ in
                 range(self.pan_blocks0)])

        def one_conv(in_ch, out_ch, ks):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, ks),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU())

        self.encode_filter = simple_net(self.band, self.endmember_num, kernelsize=1)
        self.decode_filter = basic_net(self.endmember_num, self.band, kernelsize=1)

        self.up_res4 = res_dense_net(hs_size, pan_size, endmember_num=self.endmember_num)
        self.up_res16 = res_dense_net(hs_size*int(np.sqrt(self.ratio)), pan_size, endmember_num=self.endmember_num)

        # attention blocks
        self.abun_process_net = nn.ModuleList(
            [pan_dict(endmember_num, pan_dim) for _ in range(self.abun_block_num)])

        # last convolution in every attention block
        self.conv_temp0_list = nn.ModuleList(
            [one_conv(self.endmember_num, self.endmember_num, 1) for _ in range(self.abun_block_num)])

        # part 3 last
        self.dr_layer2 = simple_net(self.abun_block_num * self.endmember_num, self.endmember_num,
                                   kernelsize=kernelsize_temp)  # 降维

    def encode_forward(self, images_hs, dim_band=1):
        return self.encode_filter(images_hs)

    def abun_pan_pro(self, abun, pan, i2, dim_band=1):
        temp0 = (self.abun_process_net[i2])[0](self.w1, self.w2, self.b1, self.b2, abun, pan)
        if self.pan_blocks0 > 1:
            for i0 in range(1, self.pan_blocks0):
                temp0 = (self.abun_process_net[i2])[i0](self.w1, self.w2, self.b1, self.b2, temp0, pan)

        return self.conv_temp0_list[i2](temp0) + abun

    def abun_pan_pro1(self, abun, pan, dim_band=1):
        temp1 = self.abun_pan_pro(abun, pan, 0)
        result = temp1
        for i2 in range(1, self.abun_block_num):
            temp1 = self.abun_pan_pro(temp1, pan, i2)
            result = torch.cat((result, temp1), dim_band)

        return self.dr_layer2(result)

    def decode_forward(self, abun_last, dim_band=1):  # 卷积成融合高光谱影像
        #  decode to spectral
        return self.decode_filter(abun_last)

    def forward(self, images_hs, images_pan, dim_band=1):

        # part 1
        abun = self.encode_forward(images_hs)

        # part 2
        abun4 = self.up_res4(self.w1, self.w2, self.b1, self.b2, abun, images_pan) 
        abun16 = self.up_res16(self.w1, self.w2, self.b1, self.b2, abun4, images_pan) + self.upsample16(abun)

        # part 3
        abun_last = self.abun_pan_pro1(abun16, images_pan) + self.upsample16(abun)

        # part 4
        hrhs = self.decode_forward(abun_last)

        return hrhs

