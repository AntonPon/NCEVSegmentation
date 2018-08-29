import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(), )
        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU, )

    def forward(self, x):
        output = self.conv1(x)
        return self.conv2(output)

class UpConvBlock(nn.Module):
    def     __init__(self, in_channles, out_channels):
        super(UpConvBlock, self).__init__()
        self.conv = unetConv2(in_size, out_size, False)
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        offset_half = offset // 2
        padding = 2 * [offset_half, offset_half]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))



class Unet(nn.Module):
    def __init__(self, n_classes=19, in_channels=3):
        super(Unet, self).__init__()
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.channels = [64, 128, 256, 512, 1024]


        self.res1 = ResBlock(self.in_channels, self.channels[0])
        self.res2 = ResBlock(self.channels[0], self.channels[1])
        self.res3 = ResBlock(self.channels[1], self.channels[2])
        self.res4 = ResBlock(self.channels[2], self.channels[3])
        self.res5 = ResBlock(self.channels[3], self.channels[4])

        self.up_concat4 = UpConvBlock(self.channels[4], self.channels[3])
        self.up_concat3 = UpConvBlock(self.channels[3], self.channels[2])
        self.up_concat2 = UpConvBlock(self.channels[2], self.channels[1])
        self.up_concat1 = UpConvBlock(self.channels[1], self.channels[0])

        self.final = nn.Conv2d(filters[0], n_classes, 1)


    def forward(self, x):
