
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



def cross_entropy2d(input, target, weight=None, size_average=True, device='cpu'):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()
    """
    # Handle inconsistent size between input and target
    if h > ht and w > wt: # upsample labels
        target = target.unsequeeze(1)
        target = F.upsample(target, size=(h, w), mode='nearest')
        target = target.sequeeze(1)
    elif h < ht and w < wt: # upsample images
        input = F.upsample(input, size=(ht, wt), mode='bilinear')
    elif h != ht and w != wt:
        raise Exception("Only support upsampling")
    """
    log_p1 = F.log_softmax(input, dim=1)
    #log_p = log_p1.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    #log_p = log_p[target.view(-1, 1).repeat(1, c) >= 0]
    #log_p = log_p.view(-1, c)

    mask = target >= 0
    target1 = target[mask]
    #loss = F.nll_loss(log_p, target1, ignore_index=250,
    #                  weight=weight, size_average=False)
    loss = F.nll_loss(log_p1, target, ignore_index=250,
                      weight=weight, size_average=False)

    # print(loss.item())
    if size_average:
        type_n = torch.cuda.FloatTensor
        if device == 'cpu':
            type_n = torch.FloatTensor
        loss = loss / mask.data.sum().type(type_n)
    return loss
