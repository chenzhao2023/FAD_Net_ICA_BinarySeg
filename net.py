import torch
from torch import nn
from torch.nn import functional as F, Conv2d
from LFDM import LFDM
from MLSA import *

class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv_Block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


class Downsample(nn.Module):
    def __init__(self, channel):
        super(Downsample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 2, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


class Upsample(nn.Module):
    def __init__(self, channel):
        super(Upsample, self).__init__()
        self.layer = nn.Conv2d(channel, channel // 2, 1, 1)

    def forward(self, x, feature_map):
        up = F.interpolate(x, scale_factor=2, mode='nearest')
        out = self.layer(up)
        return torch.cat((out, feature_map), dim=1)


class FAD_Net(nn.Module):
    def __init__(self):
        super(FAD_Net, self).__init__()
        self.c1 = Conv_Block(1, 64)
        self.d1 = Downsample(64)
        self.c2 = Conv_Block(64, 128)
        self.d2 = Downsample(128)
        self.c3 = Conv_Block(128, 256)
        self.d3 = Downsample(256)
        self.c4 = Conv_Block(256, 512)
        self.d4 = Downsample(512)
        self.c5 = Conv_Block(512, 1024)

        self.u1 = Upsample(1024)
        self.m1 = MLSA(1024, bias=False)
        self.l1 = LFDM(1024, 1024)
        self.c6 = Conv_Block(1024, 512)

        self.u2 = Upsample(512)
        self.m2 = MLSA(512, bias=False)
        self.l2 = LFDM(512, 512)
        self.c7 = Conv_Block(512, 256)

        self.u3 = Upsample(256)
        self.m3 = MLSA(256, bias=False)
        self.l3 = LFDM(256, 256)
        self.c8 = Conv_Block(256, 128)

        self.u4 = Upsample(128)
        self.m4 = MLSA(128, bias=False)
        self.l4 = LFDM(128, 128)
        self.c9 = Conv_Block(128, 64)

        self.out = Conv2d(64, 1, 3, 1, 1)
        self.TH = nn.Sigmoid()

    def forward(self, x):
        R1 = self.c1(x)
        R2 = self.c2(self.d1(R1))
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        R5 = self.c5(self.d4(R4))

        O1 = self.u1(R5, R4)
        O1 = self.m1(O1)
        O1 = self.l1(O1)
        O1 = self.c6(O1)

        O2 = self.u2(O1, R3)
        O2 = self.m2(O2)
        O2 = self.l2(O2)
        O2 = self.c7(O2)

        O3 = self.u3(O2, R2)
        O3 = self.m3(O3)
        O3 = self.l3(O3)
        O3 = self.c8(O3)

        O4 = self.u4(O3, R1)
        O4 = self.m4(O4)
        O4 = self.l4(O4)
        O4 = self.c9(O4)

        return self.TH(self.out(O4))


if __name__ == '__main__':
    x = torch.randn(2, 1, 512, 512)  # 输入为黑白图像
    net = FAD_Net()
    print(net(x).shape)  # 输出形状
