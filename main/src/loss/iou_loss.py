import torch


def intersect(segment):
        return segment.sum(-1).sum(-1)


def count_iou(flat_pred, flat_ground):
    epsilon = 1e-3
    # print(flat_pred.type(), 'flat_pred')
    # print(flat_ground.type(), 'flat_ground')
    inters = intersect(flat_pred * flat_ground)
    diff = (inters + epsilon) / \
           (intersect(flat_pred) + intersect(flat_ground) - inters + epsilon)
    return diff


def iou(segmented, ground_img, classes=19):
    result_iou = torch.zeros(classes).type(torch.cuda.FloatTensor)
    for class_c in range(classes):
        current_map = (ground_img == class_c).type(torch.cuda.FloatTensor)
        result_iou[class_c] = count_iou(segmented[class_c].view(-1), current_map.view(-1))
    return  result_iou


def mini_batch_iou(segmented, ground_img, classes=19):
    result = 0.
    for i in range(segmented.size(0)):
        result += torch.mean(1 - iou(segmented[i], ground_img[i], classes))
    return result / segmented.size(0)
