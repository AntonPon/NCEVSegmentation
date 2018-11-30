import os
import torch
import numpy as np
import scipy.misc as m
import cv2

from torch.utils.data import Dataset, DataLoader

from main.src.utils.util import recursive_glob
from main.src.utils.augmentation import Compose, RandomHorizontallyFlip, RandomRotate



class СityscapesLoader(Dataset):
    """cityscapesLoader

    https://www.cityscapes-dataset.com

    Data is derived from CityScapes, and can be downloaded from here:
    https://www.cityscapes-dataset.com/downloads/

    Many Thanks to @fvisin for the loader repo:
    https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/cityscapes.py
    """
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

    mean_rgb = {'pascal': [103.939, 116.779, 123.68], 'cityscapes': [73.15835921, 82.90891754,
                                                                     72.39239876]}  # pascal mean for PSPNet and ICNet pre-trained model

    def __init__(self, root, add_source, split="train", is_transform=False,
                 img_size=(512, 1024), augmentations=None, img_norm=True, version='pascal', step=1):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations
        """
        self.root = root
        self.split = split
        self.add_source = add_source
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.n_classes = 19
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array(self.mean_rgb[version])
        self.files = {}
        self.step = step
        self.images_base = os.path.join(self.root, 'leftImg8bit', self.split)
        self.annotations_base = os.path.join(self.root, 'gtFine', self.split)
        lister = recursive_glob(rootdir=self.images_base, suffix='.png')
        remove_ls = ['zurich_000070_000019', 'munster_000050_000019', 'bremen_000289_000019', 'munich_000383_000019']
        for el in lister:
            if remove_ls[0] in el or remove_ls[1] in el or remove_ls[2] in el or remove_ls[3] in el:
                lister.remove(el)

        self.files[split] = lister

        # self.files[split] = recursive_glob(rootdir=self.images_base, suffix='.png')

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', \
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', \
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                            'motorcycle', 'bicycle']

        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(19)))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def get_prev_img(self, img_path, distance, additional_path, split):
        path_parts = img_path.split('/')
        elements = path_parts[-1].split('_')
        prev_index = str(int(elements[-2]) - distance)
        fill_zero = (6 - len(prev_index)) * '0'
        elements[-2] = '{}{}'.format(fill_zero, prev_index)

        return os.path.join(additional_path, split, elements[0], '_'.join(elements))


    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        img_path = self.files[self.split][index].rstrip()
        distance = np.random.randint(1, 11)  # self.step

        img_path_prev = self.get_prev_img(img_path, distance, self.add_source, self.split)
        lbl_path = os.path.join(self.annotations_base,
                                img_path.split(os.sep)[-2],
                                os.path.basename(img_path)[:-15] + 'gtFine_labelIds.png')

        img = cv2.imread(img_path)
        img_prev = cv2.imread(img_path_prev)
        if img_prev is None or img is None:
            print(img_path)
            print(img_path_prev, 'break_down')

        img = np.array(img, dtype=np.uint8)
        img_prev = np.array(img_prev, dtype=np.uint8)
        lbl = m.imread(lbl_path)
        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))


        if self.augmentations is not None:
            img_prev, _ = self.augmentations(img_prev)
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img_prev, _ = self.transform(img_prev)
            img, lbl = self.transform(img, lbl)
        # return img, img_next, lbl
        return (img, img_prev, lbl, distance)


    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
            #    print("WARN: resizing labels yielded fewer classes")

        return mask

    def transform(self, img, lbl=None):
        """transform

        :param img:
        :param lbl:
        """
        img = m.imresize(img, (self.img_size[0], self.img_size[1]))  # uint8 with RGB mode
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean
        if self.img_norm:
            # Resize scales images from 0 to 255, thus we need
            # to divide by 255.0
            img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        if lbl is not None:
            classes = np.unique(lbl)
            lbl = lbl.astype(float)
            lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), 'nearest', mode='F')
            lbl = lbl.astype(int)

            # if not np.all(classes == np.unique(lbl)):
            if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
                print('after det', classes, np.unique(lbl))
                raise ValueError("Segmentation map contained invalid class values")
            lbl = torch.from_numpy(lbl).long()
        img = torch.from_numpy(img).float()

        return img, lbl


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

import os
from matplotlib import pyplot as plt


def get_image_id(str_i):
    return '_'.join(str_i.split('/')[-1].split('_')[:3])
"""
if __name__ == '__main__':
    transforms = Compose([
        RandomHorizontallyFlip(),
        RandomRotate(5)])
    img_size = None
    root_dir = 'data/images'
    steps = [1, 2, 3, 4, 5, 10, 15, 19]
    if not os.path.exists(root_dir):
       os.mkdir(root_dir)
    for step in steps:
        subfolder_path = os.path.join(root_dir, 'step_{}'.format(step))
        if not os.path.exists(subfolder_path):
            os.mkdir(subfolder_path)
        dataset = СityscapesLoader('/../../../../data/anpon/cityscapes', add_source='/../../../../data/anpon/cityscapes2/leftImg8bit_sequence',
                                   augmentations=transforms, img_size=(256, 256), is_transform=True, step=step, split='val')
        val_loader = DataLoader(dataset, batch_size=1, num_workers=2, shuffle=True)



        #for i, key_f, current_f, mask, key_name, current_name, lbl_name in enumerate(val_loader):
        for i, alpha in enumerate(val_loader):
            print(alpha[0].shape, alpha[1].shape, alpha[2].shape)
            '''
            key_f = alpha[0][0]
            current_f = alpha[1][0]
            mask = alpha[2][0]
            key_name = get_image_id(alpha[3][0])
            current_name = get_image_id(alpha[4][0])
            mask_name = get_image_id(alpha[5][0])

            img_size = (512, 1024)
            plt.subplot(311)

            plt.imshow(m.imresize(key_f, img_size))
            plt.xlabel('key_frame_{} '.format(key_name))

            plt.subplot(312)
            plt.imshow(m.imresize(current_f, img_size))
            plt.xlabel('ground_truth_{}'.format(current_name))

            plt.subplot(313)
            plt.imshow(m.imresize(mask, img_size))
            plt.xlabel('ground_truth_{}'.format(mask_name))
            plt.tight_layout()
            plt.savefig(os.path.join(subfolder_path, 'image_{}.png'.format(i)), bbox_inches='tight')
            plt.close()
            if i == 20:
                break'''

"""


def get_data_loader(root_data_path, additional_path, transforms, img_size, batch_size=32, worker_num=8, step=1):
    if len(img_size) == 1:
        img_size = 2 * img_size
    print(root_data_path)

    dataloader_trn = СityscapesLoader(root_data_path, additional_path, img_size=img_size, is_transform=True,
                                      augmentations=transforms, step=step)
    dataloader_val = СityscapesLoader(root_data_path, additional_path, img_size=img_size, is_transform=True,
                                      augmentations=transforms, split='val', step=step)

    val_loader = DataLoader(dataloader_val, batch_size=batch_size, num_workers=worker_num)
    train_loader = DataLoader(dataloader_trn, batch_size=batch_size, shuffle=True, num_workers=worker_num)
    return val_loader, train_loader
