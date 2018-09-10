import torch
from torch import nn



class IoU(nn.Module):
    def __init__(self, epsilon=1e-6):
        print(epsilon)
        super(IoU, self).__init__()
        self.epsilon = epsilon

    def intersect(self, segment):
        return segment.sum(-1).sum(-1)

    def forward(self, segmented, ground_img):
        segmented = torch.clamp(segmented, min=0)
        inters = self.intersect(segmented * ground_img)
        #print(inters)

        diff = (inters + self.epsilon) / \
               (self.intersect(segmented) + self.intersect(ground_img) - inters + self.epsilon)
        #diff = self.jaccard(segmented, ground_img)
        #print(diff)
        return torch.mean(1 - diff)


