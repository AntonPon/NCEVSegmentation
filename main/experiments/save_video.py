from matplotlib import pyplot as plt
from matplotlib import animation

import os
import cv2
import torch
import numpy as np
from main.src.models.nvce_model import NVCE
from main.src.models.unet_model import Unet
from main.src.utils.util import recursive_glob


os.environ['CUDA_VISIBLE_DEVICES'] = '2'

class iterate_frames(object):

    def recursive_glob(self, rootdir='.', suffix=''):
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

    def __init__(self, frames_folder, image_number=-1):
        self.image_names = recursive_glob(frames_folder)
        if image_number != -1:
            self.image_names = self.image_names[: image_number]
        self.total_number = len(self.image_names)
        self.current_num = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.current_num < self.total_number:
            self.current_num += 1
            return self.image_names[self.current_num - 1]
        else:
            raise StopIteration()




def prepare_image(img, img_size=(256, 256)):
    img = np.array(img, dtype=np.uint8)
    img = transform(img.copy(), img_size=img_size)
    return img


def transform(img, img_size):
    mean = [73.15835921, 82.90891754, 72.39239876]
    img = cv2.resize(img, (img_size[0], img_size[1]))  # uint8 with RGB mode
    img = img[:, :, ::-1]  # RGB -> BGR
    img = img.astype(np.float64)
    img -= mean

    img = img.astype(float) / 255.0
    # NHWC -> NCHW
    img = img.transpose(2, 0, 1)
    img = np.array([img])
    img = torch.from_numpy(img).float()
    return img

def decode_segmap(temp, n_classes=19):
    colors = [  # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]]

    label_colours = dict(zip(range(19), colors))
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, n_classes):
        r[temp == l] = label_colours[l][0]
        g[temp == l] = label_colours[l][1]
        b[temp == l] = label_colours[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb




def save_mask(path_folder, mask, mask_unet, input, frame_number):
    output = decode_segmap(mask.max(1)[1].cpu().numpy()[0])
    output_unet = decode_segmap(mask_unet.max(1)[1].cpu().numpy()[0])
    fig = plt.figure()
    plt.subplot(131)
    plt.imshow(output_unet)
    #cv2.imwrite(os.path.join(path_folder, 'img_{}.png'.format(frame_number)), output)
    plt.subplot(132)
    plt.imshow(output)
    plt.subplot(133)
    plt.imshow(input)
    plt.savefig(os.path.join(path_folder, '{}.png'.format(frame_number)))
    plt.close(fig)





if __name__ == '__main__':
    save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    save_dir_root = os.path.join(save_dir_root, '..', '..')
    path_to_model = os.path.join(save_dir_root, 'unet_cityscapes_best_model_nvce.pkl')
    path_to_unet = os.path.join(save_dir_root, 'main', 'src', 'train', 'unet_cityscapes_best_model_iou_3.pkl')
    save_path = os.path.join(save_dir_root, '../../../data/anpon/video')

    data_root = os.path.join(save_dir_root, '../../../data/anpon/cityscapes2/leftImg8bit_sequence/')
    print(data_root)
    folder_type = 'val'
    city = 'berlin'
    for i in iterate_frames(data_root, 10):
        print(i)

'''
if __name__ == '__main__':
    save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    save_dir_root = os.path.join(save_dir_root, '..', '..')
    path_to_model = os.path.join(save_dir_root, 'unet_cityscapes_best_model_nvce.pkl')
    path_to_unet = os.path.join(save_dir_root, 'main', 'src', 'train', 'unet_cityscapes_best_model_iou_3.pkl')
    save_path = os.path.join(save_dir_root, '../../../data/anpon/video')

    data_root = os.path.join(save_dir_root, '../../../data/anpon/cityscapes2/leftImg8bit_sequence/')
    folder_type = 'val'
    city = 'berlin'
    start_from = 1342
    images = recursive_glob(data_root)
    images = images[: 1000]
    model_unet = torch.nn.DataParallel(Unet())
    model = NVCE(torch.nn.DataParallel(Unet()))

    if os.path.isfile(path_to_model):
        checkpoint = torch.load(path_to_model)
        model.load_state_dict(checkpoint['model_state'])
        print('model was uploaded')
    else:
        print(path_to_model + ' was not found')

    if os.path.isfile(path_to_unet):
        checkpoint = torch.load(path_to_unet)
        model_unet.load_state_dict(checkpoint['model_state'])
        print('model unet was uploaded')
    else:
        print(path_to_unet + ' was not found')

    device = 'cuda:0'
    model.eval()
    model_unet.eval()
    with torch.no_grad():
        for i, image_path in enumerate(images):
            img = cv2.imread(image_path)
            image = prepare_image(img)
            image = image.to(device)
            output = None
            output_unet = model_unet(image)
            if (i + start_from) % 2 == 0:
                output = model(image)
            else:
                output = model(image, False)
            save_mask(save_path, output, output_unet, img, i + start_from)
            print('{} out of {}'.format(i + start_from, len(images)))
'''