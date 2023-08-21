from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import os.path as osp
import shutil
import random
import time

import torch
import torch.nn as nn
from torch.optim.adamax import Adamax
from torch.optim.adadelta import Adadelta

import src
from src import utils
from src.data import build_train_loader
from src.data import build_val_loader

# from src.data import build_otf_train_loader
# from src.matlab import matlab_speedy

logger = logging.getLogger(__name__)

import time
time.sleep(0)

Detection_target = 'mipod0.1'
Name_model = 'sia'

net = src.models.KeNet().cuda()

print(net)
num_params = sum(p.numel() for p in net.parameters())
print("Total parameters: ", num_params)


criterion_1 = nn.CrossEntropyLoss().cuda()
criterion_2 = src.models.ContrastiveLoss(margin=1.0).cuda()


def preprocess_data(images, labels, random_crop):
    # images of shape: NxCxHxW
    if images.ndim == 5:  # 1xNxCxHxW
        images = images.squeeze(0)
        labels = labels.squeeze(0)
    h, w = images.shape[-2:]

    if random_crop:
        ch = random.randint(h * 3 // 4, h)  # h // 2      #256
        cw = random.randint(w * 3 // 4, w)  # square ch   #256
        # ch = 256  # h // 2      #256
        # cw = 256  # square ch   #256

        h0 = random.randint(0, h - ch)  # 128
        w0 = random.randint(0, w - cw)  # 128
    else:
        ch, cw, h0, w0 = h, w, 0, 0

    if Name_model == 'sia':
        cw = cw & ~1
        inputs = [
            images[..., h0:h0 + ch, w0:w0 + cw // 2],
            images[..., h0:h0 + ch, w0 + cw // 2:w0 + cw]
        ]
    else:
        inputs = [images[..., h0:h0 + ch, w0:w0 + cw]]

    inputs = [x.cuda() for x in inputs]
    labels = labels.cuda()
    return inputs, labels


ckpt_dir = 'D:\shiyao_DataSet\复现其他两个网络\exp\sia\mipod0.1'
test_cover_dir = 'D:\shiyao_DataSet\Dataset\BOSSBase_256_bilinear\EXP\\test\mipod0.1\\0'
test_stego_dir = 'D:\shiyao_DataSet\Dataset\BOSSBase_256_bilinear\EXP\\test\mipod0.1\\1'


# 测试程序
def project_test():
    logger.info('Final Test=====================================================')
    # 加载数据集
    test_loader = build_val_loader(
        test_cover_dir, test_stego_dir, batch_size=32,
        num_workers=0)
    # 加载最佳模型
    best_ckpt = os.path.join(ckpt_dir, 'model_best.pth.tar')
    net.load_state_dict(torch.load(best_ckpt)['state_dict'])

    net.eval()
    test_loss = 0.
    test_accuracy = 0.
    test_time1 = time.time()
    test_time = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = preprocess_data(data['image'], data['label'], False)
            iter_time = time.time()
            outputs, feats_0, feats_1 = net(*inputs)
            _, argmax = torch.max(outputs, 1)
            test_time += time.time() - iter_time
            test_accuracy += src.models.accuracy(outputs, labels).item()
            test_loss += criterion_1(outputs, labels).item() + 0.1 * criterion_2(feats_0, feats_1, labels)

    test_loss /= len(test_loader)
    test_accuracy /= len(test_loader)

    print('Final Test set: Loss: {:.6f}, Accuracy: {:.2f}%'.format(
        test_loss, 100 * test_accuracy))
    print('Final Test Time: {:.6f}s'.format(test_time))
    print('Final Test Time: {:.6f}s'.format(time.time() - test_time1))


project_test()

