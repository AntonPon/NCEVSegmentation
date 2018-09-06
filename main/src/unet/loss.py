import torch
from torch import nn


class IoU(nn.Module):
    def __init__(self, epsilon=1e6):
        super(IoU, self).__init__()
        self.epsilon = epsilon

    def intersect(self, segment_one, segment_two):
        return (segment_one * segment_two).sum(-1).sum(-1)

    def forward(self, segmented, ground_img, weights):
        segmented = segmented[0]
        ground_img = ground_img[0]

        inters = self.intersect(segmented, ground_img)

        diff = (inters + self.epsilon) / \
               (self.intersect(segmented, segmented) + self.intersect(ground_img, ground_img)- inters + self.epsilon) * weights
        print(diff)
        return torch.mean(diff)
