'''
Misc Utility functions
'''
import os
import torch


def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    print(suffix)
    images = [os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames if filename.endswith(suffix)]
    images.sort()
    return images


def add_info(writer, epoch, loss=0, miou=0, mode='train'):
    writer.add_scalar('loss/{}'.format(mode), loss, epoch)
    writer.add_scalar('miou/{}'.format(mode), miou, epoch)


def add_image(writer, epoch, images):
    writer.add_image('test images', images, epoch)


def save_model(epoch, model_state, optimizer_state, model='fpn', dataset='cityscapes'):
    state = {'epoch': epoch + 1,
             'model_state': model_state,
             'optimizer_state': optimizer_state, }
    torch.save(state, "{}_{}_best_model_iou.pkl".format(model, dataset))
