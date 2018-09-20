
from torch import nn
import main.src.unet.lovazs_loss as lovasz



class IoU(nn.Module):
    def __init__(self, epsilon=1e-3):
        super(IoU, self).__init__()
        self.epsilon = epsilon
    '''
    def intersect(self, segment):
        return segment.sum(-1).sum(-1)

    def iou(self, segmented, ground_img):
        inters = self.intersect(segmented * ground_img)

        diff = (inters + self.epsilon) / \
               (self.intersect(segmented) + self.intersect(ground_img) - inters + self.epsilon)
        return diff
'''
    def forward(self, segmented, ground_img):
        return lovasz.lovasz_softmax(segmented, ground_img, ignore=250, per_image=True)
