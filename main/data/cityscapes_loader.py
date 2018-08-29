import os
import cv2
import torch
from torch.utils.data import Dataset


class CityscapesDataset(Dataset):
    """cityscapes dataset"""

    def __init__(self, root, mode='train', transform=None):
        """
        :param root (string): Path to the data gtFine and left8Image.
        :param mode (string): train/val upload images.
        :param transform (callable, optional):  Optional transform to be appliedcv on a sample.
        """

        self.root = root
        self.mode = mode
        self.transform = transform

        # support data
        self.n_classes = 19
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', \
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', \
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                            'motorcycle', 'bicycle']

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass