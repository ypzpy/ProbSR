import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as sn
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm.std import trange
from torch.utils.data import dataloader
    

class ResidualLearning(torch.nn.Module):
    def __init__(self):
        super(ResidualLearning, self).__init__()
        self.layer1 = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1,padding=1)
        self.layer2 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1,padding=1)
        self.layer4 = torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1,padding=1)
        self.layer3 = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1,padding=1)
        self.PReLU = torch.nn.PReLU()
    def forward(self, x):
        x = self.layer1(x)
        x = self.PReLU(x)
        x = self.layer2(x)
        x = self.PReLU(x)
        x = self.layer4(x)
        x = self.PReLU(x)
        x = self.layer3(x)
        return x
    

class DownScale3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pool_size=2):
        super(DownScale3D, self).__init__()
        self.layer1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.layer2 = nn.Conv3d(out_channels, out_channels*2, kernel_size=kernel_size, stride=stride, padding=padding)
        self.layer7 = nn.Conv3d(out_channels*2, out_channels*2, kernel_size=kernel_size, stride=stride, padding=padding)
        self.layer3 = nn.MaxPool3d(kernel_size=pool_size, stride=pool_size)
        self.layer4 = nn.Conv3d(out_channels*2, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.layer5 = nn.MaxPool3d(kernel_size=pool_size, stride=pool_size)
        self.layer6 = nn.Conv3d(out_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.PReLU = torch.nn.PReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.PReLU(x)
        x = self.layer2(x)
        x = self.PReLU(x)
        x = self.layer7(x)
        x = self.PReLU(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.PReLU(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x
    

# 上采样+拼接
class Up(nn.Module):
    def __init__(self,in_channels,out_channels,bilinear=True):
        '''
        :param in_channels: 输入通道数
        :param out_channels:  输出通道数
        :param bilinear: 是否采用双线性插值，默认采用
        '''
        super(Up, self).__init__()
        if bilinear:
            # 双线性差值
            self.up = nn.Upsample(scale_factor=2,mode='trilinear',align_corners=True)
            self.conv = doubleConv(in_channels,out_channels,in_channels//2) # 拼接后为1024，经历第一个卷积后512
        else:
            # 转置卷积实现上采样
            # 输出通道数减半，宽高增加一倍
            self.up = nn.ConvTranspose3d(in_channels,out_channels//2,kernel_size=2,stride=2)
            self.conv = doubleConv(in_channels,out_channels)

    def forward(self,x1,x2):
        # 上采样
        x1 = self.up(x1)
        # 拼接
        x = torch.cat([x1,x2],dim=1)
        # 经历双卷积
        x = self.conv(x)
        return x

# 双卷积层
def doubleConv(in_channels,out_channels,mid_channels=None):
    '''
    :param in_channels: 输入通道数 
    :param out_channels: 双卷积后输出的通道数
    :param mid_channels: 中间的通道数，这个主要针对的是最后一个下采样和上采样层
    :return: 
    '''
    if mid_channels is None:
        mid_channels = out_channels
    layer = []
    layer.append(nn.Conv3d(in_channels,mid_channels,kernel_size=3,padding=1,bias=False))
    layer.append(nn.BatchNorm3d(mid_channels))
    layer.append(nn.ReLU(inplace=True))
    layer.append(nn.Conv3d(mid_channels,out_channels,kernel_size=3,padding=1,bias=False))
    layer.append(nn.BatchNorm3d(out_channels))
    layer.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layer)

# 下采样
def down(in_channels,out_channels):
    # 池化 + 双卷积
    layer = []
    layer.append(nn.MaxPool3d(2,stride=2))
    layer.append(doubleConv(in_channels,out_channels))
    return nn.Sequential(*layer)

# 整个网络架构
class U_net(nn.Module):
    def __init__(self,in_channels,out_channels,bilinear=True,base_channel=64):
        '''
        :param in_channels: 输入通道数，一般为3，即彩色图像
        :param out_channels: 输出通道数，即网络最后输出的通道数，一般为2，即进行2分类
        :param bilinear: 是否采用双线性插值来上采样，这里默认采取
        :param base_channel: 第一个卷积后的通道数，即64
        '''
        super(U_net, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        # 输入
        self.in_conv = doubleConv(self.in_channels,base_channel) # 1,64
        # 下采样
        self.down1 = down(base_channel,base_channel*2) # 64,128
        self.down2 = down(base_channel*2,base_channel*4) #128,256
        # 最后一个下采样，通道数不翻倍（因为双线性差值，不会改变通道数的，为了可以简单拼接，就不改变通道数）
        # 当然，是否采取双线新差值，还是由我们自己决定
        factor = 2  if self.bilinear else 1
        self.down3 = down(base_channel*4,base_channel*8 // factor) # 256,256
        # 上采样 + 拼接
        self.up1 = Up(base_channel*8 ,base_channel*4 // factor,self.bilinear) 
        self.up2 = Up(base_channel*4 ,base_channel*2 // factor,self.bilinear)
        self.up3 = Up(base_channel*2 ,base_channel,self.bilinear)
        # 输出
        self.out = nn.Conv3d(in_channels=base_channel,out_channels=self.out_channels,kernel_size=1)

    def forward(self,x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # 不要忘记拼接
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        out = self.out(x)

        return out