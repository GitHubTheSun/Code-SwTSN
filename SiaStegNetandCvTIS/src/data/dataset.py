from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp

import numpy as np
from torch.utils.data import Dataset
from PIL import Image
# import cv2

# from ..matlab import S_UNIWARD


class CoverStegoDataset(Dataset):

    def __init__(self, cover_dir, stego_dir, transform=None):
        self._transform = transform

        self.images, self.labels = self.get_items(cover_dir, stego_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.images[idx]))
        image = np.expand_dims(image, 2)  # (H, W, C)
        assert image.ndim == 3

        sample = {
            'image': image,
            'label': self.labels[idx]
        }

        if self._transform:
            sample = self._transform(sample)
        return sample

    @staticmethod
    def get_items(cover_dir, stego_dir):
        images, labels = [], []

        cover_names = sorted(os.listdir(cover_dir))
        if stego_dir is not None:
            stego_names = sorted(os.listdir(stego_dir))
            assert cover_names == stego_names

        file_names = cover_names
        if stego_dir is None:
            dir_to_label = [(cover_dir, 0), ]
        else:
            dir_to_label = [(cover_dir, 0), (stego_dir, 1)]
        for image_dir, label in dir_to_label:
            for file_name in file_names:
                image_path = osp.join(image_dir, file_name)
                if not osp.isfile(image_path):
                    raise FileNotFoundError('{} not exists'.format(image_path))
                images.append(image_path)
                labels.append(label)

        return images, labels


class DatasetPair(Dataset):
    def __init__(self, cover_dir, stego_dir, transform):
        self.cover_dir = cover_dir
        self.stego_dir = stego_dir
        # [x.split('/')[-1] for x in glob(cover_dir + '/*')]
        self.cover_list = os.listdir(cover_dir)
        # print(self.cover_list)
        self.transform = transform
        assert len(self.cover_list) != 0, "cover_dir is empty"
        # stego_list = ['.'.join(x.split('/')[-1].split('.')[:-1])
        #               for x in glob(stego_dir + '/*')]

    def __len__(self):
        return len(self.cover_list)

    def __getitem__(self, idx):
        idx = int(idx)
        labels = np.array([0, 1], dtype='int32')

        cover_path = os.path.join(self.cover_dir, self.cover_list[idx])
        # print(cover_path)
        cover = Image.open(cover_path)
        images = np.empty((2, cover.size[0], cover.size[1], 1), dtype='uint8')
        images[0, :, :, 0] = np.array(cover)

        # print(self.cover_list[idx])
        stego_path = os.path.join(self.stego_dir, self.cover_list[idx])
        # print(stego_path)
        stego = Image.open(stego_path)
        images[1, :, :, 0] = np.array(stego)

        samples = {'images': images, 'labels': labels}
        if self.transform:
            samples = self.transform(samples)
        return samples
