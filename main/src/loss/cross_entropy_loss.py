
import torch
import torch.nn as nn
import torch.nn.functional as F

import main.src.loss.lovazs_loss as lovasz


class IoU(nn.Module):
    def __init__(self, epsilon=1e-3):
        super(IoU, self).__init__()
        self.epsilon = epsilon

    def forward(self, segmented, ground_img):
        return lovasz.lovasz_softmax(segmented, ground_img, ignore=250, per_image=True)


def cross_entropy2d(input, target, weight=None, size_average=True, device='cpu', frame_distance=None):
    log_p1 = F.log_softmax(input, dim=1)
    mask = target >= 0
    loss = F.nll_loss(log_p1, target, ignore_index=250,
                      weight=weight, size_average=False, reduce=False, reduction='none')

    dim_size = (1, 2)
    loss = loss.sum(dim=dim_size)
    if size_average:
        type_n = torch.cuda.FloatTensor
        if device == 'cpu':
            type_n = torch.FloatTensor
        loss = loss / mask.data.sum().type(type_n)
    if frame_distance is not None:
        loss = loss * frame_distance.to(device)
    loss = loss.sum()
    return loss


