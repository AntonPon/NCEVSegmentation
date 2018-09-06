import torch
from torch import nn



class IoU(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(IoU, self).__init__()
        self.epsilon = epsilon

    def intersect(self, segment):
        return segment.sum(-1).sum(-1)

    def forward(self, segmented, ground_img, weights):
        inters = self.intersect(segmented * ground_img)

        diff = (inters + self.epsilon) / \
               (self.intersect(segmented**2) + self.intersect(ground_img**2) - inters + self.epsilon) * weights
        return 1 - torch.mean(diff)

