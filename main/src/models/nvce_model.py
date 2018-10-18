from torch.nn import Module, Sequential
from main.src.models.unet_model import Unet

class NVCE(Module):

    def __init__(self, extractor=Unet(), n_classes=19, in_channels=3):
        super(NVCE, self).__init__()

        self.n_classes = n_classes
        self.in_channel = in_channels
        #self.extractor = extractor
        unet_layers = list(extractor.children())
        self.key_frame = None
        self.res1 = unet_layers[0]
        self.maxpool1 = unet_layers[1]
        self.res2 = unet_layers[2]
        self.maxpool2 = unet_layers[3]
        self.res3 = unet_layers[4]
        self.maxpool3 = unet_layers[5]
        self.res4 = unet_layers[6]
        self.maxpool4 = unet_layers[7]

        self.res5 = unet_layers[8]
        self.maxpool5 = unet_layers[9]
        self.res6 = unet_layers[10]

        self.up_sample0 = unet_layers[11]
        self.up_sample1 = unet_layers[12]
        self.up_sample2 = unet_layers[13]
        self.up_sample3 = unet_layers[14]
        self.up_sample4 = unet_layers[15]

        self.result = unet_layers[16]

    def forward(self, x, is_keyframe=True):
        res1 = self.res1(x)
        maxpool1 = self.maxpool1(res1)
        res2 = self.res2(maxpool1)
        maxpool2 = self.maxpool2(res2)

        res3 = self.res3(maxpool2)
        maxpool3 = self.maxpool3(res3)
        if is_keyframe:
            res4 = self.res4(maxpool3)
            maxpool4 = self.maxpool4(res4)
            res5 = self.res5(maxpool4)
            maxpool5 = self.maxpool5(res5)
            res6 = self.res6(maxpool5)

            up0 = self.up_sample0(res6, res5)
            up1 = self.up_sample1(up0, res4)
            self.key_frame = up1

        up2 = self.up_sample2(self.key_frame, res3)
        up3 = self.up_sample3(up2, res2)
        up4 = self.up_sample4(up3, res1)
        return self.result(up4)

