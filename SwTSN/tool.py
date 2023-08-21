import os
import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from PIL import Image


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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


class ToTensor(object):
    def __call__(self, samples):
        images, labels = samples['images'], samples['labels']
        # images = images.transpose((0, 3, 1, 2)).astype('float32')
        images = (images.transpose((0, 3, 1, 2)).astype('float32') / 255)
        return {'images': torch.from_numpy(images),
                'labels': torch.from_numpy(labels).long()}


class AugData():
    def __call__(self, samples):
        images, labels = samples['images'], samples['labels']

        # Rotation
        rot = np.random.randint(0, 3)
        images = np.rot90(images, rot, axes=[1, 2]).copy()
        # Mirroring
        if np.random.random() < 0.5:
            images = np.flip(images, axis=2).copy()

        new_sample = {'images': images, 'labels': labels}

        return new_sample


def _label_sum(pred, target):
    pred = pred.view_as(target)
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    l1 = []
    for i in range(len(target)):
        l1.append(pred[i] + target[i])
    # l1.count(0)即为 正确被判定为载体图像（阴性）的数量。l1.count(2)，即为正确被判定为载密图像（阳性）的数量。l1.count(0)+l1.count(2) 即为判断正确的总个数
    return l1.count(0), l1.count(2), l1.count(0) + l1.count(2)