import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from res2net import res2net50_v1b_26w_4s
import numpy as np
import cv2
device = torch.device('cuda')


class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.relu = nn.ReLU()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c)
        )

    def forward(self, inputs):
        x1 = self.conv(inputs)
        x2 = self.shortcut(inputs)
        x = self.relu(x1 + x2)
        return x


class Bottleneck(nn.Module):
    def __init__(self, in_c, out_c, dim, num_layers=2):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=8)
        self.tblock = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        b, c, h, w = x.shape
        x = x.reshape((b, c, h*w))
        x = self.tblock(x)
        x = x.reshape((b, c, h, w))
        x = self.conv2(x)
        return x


class DilatedConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
 
        self.c1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )
        self.c3 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=6, dilation=6),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )
        self.c5 = nn.Sequential(
            nn.Conv2d(out_c*3, out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, inputs):
        x1 = self.c1(inputs)
        x2 = self.c2(inputs)
        x3 = self.c3(inputs)
        x = torch.cat([x1, x2, x3], axis=1)
        x = self.c5(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.r1 = ResidualBlock(in_c[0]+in_c[1], out_c)
        self.r2 = ResidualBlock(out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.r1(x)
        x = self.r2(x)
        return x


class DecoderBlock2(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.r1 = ResidualBlock(in_c[0]+in_c[1]+in_c[2], out_c)
        self.r2 = ResidualBlock(out_c, out_c)

    def forward(self, input1, input2, skip):
        x1 = self.up(input1)
        x2 = self.up(input2)
        x = torch.cat([x1, x2, skip], axis=1)
        x = self.r1(x)
        x = self.r2(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


# 空间注意力
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


# CBAM注意力模块
class CBAM(nn.Module):
    def __init__(self, c1, c2=64):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


class ECTransNet(nn.Module):
    def __init__(self):
        super().__init__()

        """ Res2Net50 """
        backbone = res2net50_v1b_26w_4s(pretrained=True)
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.layer1 = nn.Sequential(backbone.maxpool, backbone.layer1)
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.T4_T3 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512),
                                   nn.ReLU(inplace=True))
        self.T3_T2 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512),
                                   nn.ReLU(inplace=True))
        self.T4_T3_T2 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512),
                                      nn.ReLU(inplace=True))
        self.l2 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256),
                                nn.ReLU(inplace=True))
        self.l1 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
                                nn.ReLU(inplace=True))
        self.CBAM = CBAM(64)

        """ Bridge blocks """
        self.b1 = Bottleneck(1024, 256, 256, num_layers=6)
        self.b2 = DilatedConv(1024, 256)
        self.b4 = DilatedConv(512, 512)
        self.b6 = DilatedConv(256, 512)

        """ Decoder """
        self.d1 = DecoderBlock([512, 512], 256)
        self.d2 = DecoderBlock2([256, 256, 256], 128)
        self.d3 = DecoderBlock2([128, 128, 64], 64)
        self.d4 = DecoderBlock([64, 3], 32)
        self.output = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x, heatmap=None):
        # s0[16, 3, 256, 256]
        s0 = x
        s1 = self.layer0(s0)    ## [16, 64, 128(h/2), 128(w/2)]
        s2 = self.layer1(s1)    ## [16, 256, 64(h/4), 64(w/4)]
        s3 = self.layer2(s2)    ## [16, 512, 32_h/8, 32_w/8]
        s4 = self.layer3(s3)    ## [16, 1024, 16_h/16, 16_w/16]

        # edge compensation module
        b1 = self.b1(s4)# 16, 256, 16, 16
        b2 = self.b2(s4)
        T4 = torch.cat([b1, b2], axis=1)# 16，512，16，16
        b4 = self.b4(s3)
        b6 = self.b6(s2)
        T4_3 = self.T4_T3(abs(F.interpolate(T4, size=b4.size()[2:], mode='bilinear') - b4))
        T3_2 = self.T3_T2(abs(F.interpolate(b4, size=b6.size()[2:], mode='bilinear') - b6))
        T4_3_2 = self.T4_T3_T2(abs(F.interpolate(T4_3, size=T3_2.size()[2:], mode='bilinear') - T3_2))

        # feature aggregation decoder
        d1 = self.d1(T4, s3)
        l2 = self.l2(T4_3)
        d2 = self.d2(d1, l2, s2)
        l1 = self.l1(T3_2)+self.l1(T4_3_2)
        s1 = self.CBAM(s1)
        d3 = self.d3(d2, l1, s1)
        d4 = self.d4(d3, s0)

        y = self.output(d4)

        return y

if __name__ == "__main__":
    x = torch.randn((16, 3, 256, 256))
    model = ECTransNet()
    y = model(x)
    print(y.shape)
