import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(ResBlock, self).__init__()
        if batch_norm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                                       nn.BatchNorm2d(out_channels),
                                       nn.ReLU(),)
            self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                                       nn.BatchNorm2d(out_channels),
                                       nn.ReLU(),)
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                                       nn.ReLU(), )
            self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                                       nn.ReLU(), )

    def forward(self, x):
        output = self.conv1(x)
        return self.conv2(output)


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConvBlock, self).__init__()
        self.conv = ResBlock(in_channels, out_channels, False)
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, prev_up, conv_in):
        up_cur = self.up(prev_up)
        resize_input = (up_cur.size(2) - conv_in.size(2)) // 2
        padd = 2 * [resize_input, resize_input]
        resized_conv = F.pad(conv_in, padd)

        return self.conv(torch.cat([resized_conv, up_cur], 1))

class FinalBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(FinalBlock, self).__init__()
        self.conv_f = nn.Conv2d(in_channel, out_channel, stride)
       # self.softmax = nn.Softmax2d()

    def forward(self, final_input):
        #result =  self.conv_f(final_input)
        #return self.softmax(result)
        return self.conv_f(final_input)




class Unet(nn.Module):
    def __init__(self, n_classes=19, in_channels=3):
        super(Unet, self).__init__()
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.channels = [16, 32, 64, 128, 256, 512]

        self.res1 = ResBlock(self.in_channels, self.channels[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.res2 = ResBlock(self.channels[0], self.channels[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.res3 = ResBlock(self.channels[1], self.channels[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.res4 = ResBlock(self.channels[2], self.channels[3])
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.res5 = ResBlock(self.channels[3], self.channels[4])
        self.maxpool5 = nn.MaxPool2d(kernel_size=2)
        self.res6 = ResBlock(self.channels[4], self.channels[5])

        self.up_sample0 = UpConvBlock(self.channels[5], self.channels[4])
        self.up_sample1 = UpConvBlock(self.channels[4], self.channels[3])
        self.up_sample2 = UpConvBlock(self.channels[3], self.channels[2])
        self.up_sample3 = UpConvBlock(self.channels[2], self.channels[1])
        self.up_sample4 = UpConvBlock(self.channels[1], self.channels[0])

        self.result = FinalBlock(self.channels[0], self.n_classes, 1)

    def forward(self, x):
        res1 = self.res1(x)
        maxpool1 = self.maxpool1(res1)
        res2 = self.res2(maxpool1)
        maxpool2 = self.maxpool2(res2)
        res3 = self.res3(maxpool2)
        maxpool3 = self.maxpool3(res3)
        res4 = self.res4(maxpool3)
        maxpool4 = self.maxpool4(res4)

        res5 = self.res5(maxpool4)
        maxpool5 = self.maxpool5(res5)
        res6 = self.res6(maxpool5)

        up0 = self.up_sample0(res6, res5)
        up1 = self.up_sample1(up0, res4)
        up2 = self.up_sample2(up1, res3)
        up3 = self.up_sample3(up2, res2)
        up4 = self.up_sample4(up3, res1)
        return self.result(up4)

