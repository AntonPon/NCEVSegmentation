from matplotlib import pyplot as plt
from matplotlib import animation

import os
import cv2
import torch
import numpy as np
# from main.src.models.nvce_model import NVCE
# from main.src.models.unet_model import Unet
from main.src.models.fpn_model import FPN
from main.src.models.nvce_fpn_model import NVCE_FPN
from main.src.utils.util import recursive_glob


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
    plt.xlabel('fpn')
    plt.subplot(132)
    plt.imshow(output)
    plt.xlabel('nvce_fpn')
    plt.subplot(133)
    plt.imshow(input)
    plt.xlabel('input')
    plt.savefig(os.path.join(path_folder, '{}.png'.format(frame_number)))
    plt.close(fig)


def get_prev_img(img_path, distance, additional_path, split):
    path_parts = img_path.split('/')
    elements = path_parts[-1].split('_')
    prev_index = str(int(elements[-2]) - distance)
    fill_zero = (6 - len(prev_index)) * '0'
    elements[-2] = '{}{}'.format(fill_zero, prev_index)

    return os.path.join(additional_path, split, elements[0], '_'.join(elements))



if __name__ == '__main__':
    save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    save_dir_root = os.path.join(save_dir_root, '..', '..')
    path_to_model = os.path.join(save_dir_root, 'fpn_loss_cityscapes_alpha_0_3_distance_random_overfeat_model_nvce.pkl')
    path_to_unet = os.path.join(save_dir_root, 'fpn_bold_rewrite_cityscapes_best_model_iou.pkl')
    save_path = os.path.join(save_dir_root, '../../../data/anpon/video')
    start_from = 0
    data_root = os.path.join(save_dir_root, '../../../data/anpon/cityscapes2/leftImg8bit_sequence/train')
    #data_root = os.path.join(save_dir_root, '../../../data/anpon/cityscapes/leftImg8bit/train')
    folder_type = 'train'
    city = 'aachen'
    images = recursive_glob(data_root)
    images = images[: 500]
    print(images[:3])
    model_unet = FPN(num_classes=19) # torch.nn.DataParallel(Unet())
    model = NVCE_FPN() # NVCE(torch.nn.DataParallel(Unet()))

    if os.path.isfile(path_to_model):
        checkpoint = torch.load(path_to_model)
        model.load_state_dict(checkpoint['model_state'])
        print('model was uploaded')
    else:
        print(path_to_model + ' was not found')

    if os.path.isfile(path_to_unet):
        checkpoint = torch.load(path_to_unet)
        model_unet.load_state_dict(checkpoint['model_state'])
        print('model fpn was uploaded')
    else:
        print(path_to_unet + ' was not found')

    device = 'cuda:0'
    model.to(device)
    model_unet.to(device)
    model.eval()
    model_unet.eval()
    with torch.no_grad():
        for i, image_path in enumerate(images):
            img = cv2.imread(image_path)
            image = prepare_image(img, (512, 512))
            image = image.to(device)
            output = None # model(image)
            output_unet = model_unet(image)
            if (i) % 2 == 0:
                output = model(image)
            else:
                output = model(image, False)
            save_mask(save_path, output, output_unet, img, i)
            print('{} out of {}'.format(i, len(images)))

'''
import time
def speed_nvce():
    save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    save_dir_root = os.path.join(save_dir_root, '..', '..')
    path_to_model = os.path.join(save_dir_root, 'fpn_loss_cityscapes_alpha_0.2_distance_random_model_nvce.pkl')
    path_to_unet = os.path.join(save_dir_root, 'fpn_bold_rewrite_cityscapes_best_model_iou.pkl')

    data_root = os.path.join(save_dir_root, '../../../data/anpon/cityscapes2/leftImg8bit_sequence/')
    images = recursive_glob(data_root)
    images = images[: 1000]
    model_unet = load_network(FPN(num_classes=19), path_to_unet)  # torch.nn.DataParallel(Unet())
    model = load_network(NVCE_FPN(), path_to_model)# NVCE(torch.nn.DataParallel(Unet()))
    print('count_time for nvce: {}'.format(count_time(model, images)))
    print('count_time for fpn: {}'.format(count_time(model_unet, images, True)))


def load_network(network, path_to_load):
    model = network
    if os.path.isfile(path_to_load):
        checkpoint = torch.load(path_to_load)
        model.load_state_dict(checkpoint['model_state'])
        print('model fpn was uploaded')
    else:
        print(path_to_load + ' was not found')
    return model


def count_time(model, data=None, nvce=False):
    device = 'cuda:0'
    model.to(device)
    model.eval()
    time_count = 0
    data_img = list()
    for path_im in data:
        img = cv2.imread(path_im)
        image = prepare_image(img, (512, 512))
        data_img.append(image)
    print('start estimated time')
    with torch.no_grad():
        for i, image_r in enumerate(data_img):
            image = image_r.to(device)
            start = time.time()
            if i % 2 == 0 or nvce:
                output = model(image)
            elif not nvce:
                output = model(image, False)
            end_t = time.time()
            time_count += end_t - start
            print('{}_{}'.format(i, end_t - start))
    print('end estimated time')
    return time_count


if __name__ == '__main__':
    speed_nvce()
'''