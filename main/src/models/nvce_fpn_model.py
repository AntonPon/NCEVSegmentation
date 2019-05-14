import torch
from torch import nn
from main.src.models.fpn_model import *
from torchvision.models import densenet121


class NVCE_FPN(torch.nn.Module):

    def __init__(self, extractor=FPN(num_classes=19)):
        super(NVCE_FPN, self).__init__()
        children_layes = list(extractor.children())[1:]

        self.enc0 = children_layes[0]
        self.pool0 = children_layes[1]

        self.enc1 = children_layes[2]
        self.enc2 = children_layes[3]
        self.enc3 = children_layes[4]
        self.enc4 = children_layes[5]
        self.norm = children_layes[6]

        self.tr1 = children_layes[7]
        self.tr2 = children_layes[8]
        self.tr3 = children_layes[9]

        self.lateral4 = children_layes[10]
        self.lateral3 = children_layes[11]
        self.lateral2 = children_layes[12]
        self.lateral1 = children_layes[13]
        self.lateral0 = children_layes[14]

        self.head1 = children_layes[15]
        self.head2 = children_layes[16]
        self.head3 = children_layes[17]
        self.head4 = children_layes[18]

        self.smooth1 = children_layes[19]
        self.smooth2 = children_layes[20]

        self.final = children_layes[21]

        self.enc3_key = None
        self.enc4_key = None

    def forward(self, x, is_keyframe=True, regularization=False):
        enc0 = self.enc0(x)
        pool0 = self.pool0(enc0)

        enc1 = self.enc1(pool0)
        tr1 = self.tr1(enc1)

        enc2 = self.enc2(tr1)
        tr2 = self.tr2(enc2)


        reg = None
        if is_keyframe:
            self.enc3_key = self.enc3(tr2)
            tr3 = self.tr3(self.enc3_key)

            self.enc4_key = self.enc4(tr3)
            self.enc4_key = self.norm(self.enc4_key)
        else:
            #self.enc4_key = self.enc4_key.detach()
            #self.enc3_key = self.enc3_key.detach()
            if regularization:
                current_enc3 = self.enc3(tr2)
                current_tr3 = self.tr3(current_enc3)

                current_enc4 = self.enc4(current_tr3)
                current_enc4 = self.norm(current_enc4)

                reg_3 = current_enc3 - self.enc3_key
                reg_4 = current_enc4 - self.enc4_key
                # reg = torch.sqrt(torch.sum(reg_4 * reg_4) + torch.sum(reg_3 * reg_3)) L2
                reg = (torch.sum(torch.abs(reg_4)) + torch.sum(torch.abs(reg_3)))


        lateral4 = self.lateral4(self.enc4_key)
        lateral3 = self.lateral3(self.enc3_key)

        '''
        enc3_key = self.enc3(tr2)
        tr3 = self.tr3(enc3_key)

        reg = None
        if is_keyframe:

            self.enc4_key = self.enc4(tr3)
            self.enc4_key = self.norm(self.enc4_key)
        else:
            if regularization:

                current_enc4 = self.enc4(tr3)
                current_enc4 = self.norm(current_enc4)

                reg_4 = current_enc4 - self.enc4_key
                # reg = torch.sqrt(torch.sum(reg_4 * reg_4) + torch.sum(reg_3 * reg_3)) L2
                reg = torch.sum(torch.abs(reg_4))

        lateral4 = self.lateral4(self.enc4_key)
        lateral3 = self.lateral3(enc3_key)
        '''
        lateral2 = self.lateral2(enc2)
        lateral1 = self.lateral1(enc1)
        lateral0 = self.lateral0(enc0)

        map4 = lateral4
        map3 = lateral3 + nn.functional.upsample(map4, scale_factor=2, mode="nearest")
        map2 = lateral2 + nn.functional.upsample(map3, scale_factor=2, mode="nearest")
        map1 = lateral1 + nn.functional.upsample(map2, scale_factor=2, mode="nearest")

        map0 = lateral0

        map4 = nn.functional.upsample(self.head4(map4), scale_factor=8, mode="nearest")
        map3 = nn.functional.upsample(self.head3(map3), scale_factor=4, mode="nearest")
        map2 = nn.functional.upsample(self.head2(map2), scale_factor=2, mode="nearest")
        map1 = nn.functional.upsample(self.head1(map1), scale_factor=1, mode="nearest")

        smoothed = self.smooth1(torch.cat([map4, map3, map2, map1], dim=1))
        smoothed = nn.functional.upsample(smoothed, scale_factor=2, mode="nearest")
        smoothed = self.smooth2(smoothed + map0)
        smoothed = nn.functional.upsample(smoothed, scale_factor=2, mode="nearest")

        final = self.final(smoothed)

        if regularization:
            return final, reg.item()
        return final
if __name__ == "__main__":
    fpn = FPN(19)

    #print(list(fpn.children())[-7:])
    for child in list(fpn.children())[-7:]:
        print(child)
        print('----------------------------------------------------')
