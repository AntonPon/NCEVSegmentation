import torch
from torch import nn


class IoU(nn.Module):
    def __init__(self, epsilon=1e-6):
        print(epsilon)
        super(IoU, self).__init__()
        self.epsilon = epsilon

    def intersect(self, segment):
        return segment.sum(-1).sum(-1)

    def iou(self, segmented, ground_img):
        inters = self.intersect(segmented * ground_img)

        diff = (inters + self.epsilon) / \
               (self.intersect(segmented) + self.intersect(ground_img) - inters + self.epsilon)
        return diff

    def forward(self, segmented, ground_img):

        return torch.mean(1 - self.iou(segmented, ground_img))


