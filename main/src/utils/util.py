'''
Misc Utility functions
'''
import os

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
    writer.add_scalar('{}_loss'.format(mode), loss, epoch)
    writer.add_scalar('{}_miou'.format(mode), miou, epoch)
