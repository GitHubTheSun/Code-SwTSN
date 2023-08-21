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

# 程序延时
import time
time.sleep(0)  # second

logger = logging.getLogger(__name__)

Name_model = 'CvT_VS'


def parse_args():
    parser = argparse.ArgumentParser()
    ckpt_dir = 'D:\shiyao_DataSet\复现其他两个网络\exp\Vsize'
    ckpt_dir = os.path.join(ckpt_dir, Name_model)
    parser.add_argument('--ckpt-dir', dest='ckpt_dir', type=str, default=ckpt_dir)
    parser.add_argument('--epoch', dest='epoch', type=int, default=500)  # default=1000
    parser.add_argument('--random-crop', dest='random_crop', action='store')  # action='store_true')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=8)
    parser.add_argument('--num-workers', dest='num_workers', type=int, default=0)
    parser.add_argument('--model', dest='model', type=str, default=Name_model)
    parser.add_argument('--seed', dest='seed', type=int, default=-1)
    parser.add_argument('--log-interval', dest='log_interval', type=int, default=50)
    args = parser.parse_args()
    return args


def setup(args):
    log_file = osp.join(args.ckpt_dir, 'log.txt')
    utils.configure_logging(file=log_file, root_handler_type=1)
    utils.set_random_seed(None if args.seed < 0 else args.seed)
    logger.info('Command Line Arguments: {}'.format(str(args)))


args = parse_args()
setup(args)


logger.info('Building model')
net = src.models.CvT_VS().cuda()
optimizer = Adamax(net.parameters(), lr=0.001, eps=1e-8, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300, 400], gamma=0.1)


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
        # ch = random.randint(h * 3 // 4, h)  # h // 2      #256
        # cw = random.randint(w * 3 // 4, w)  # square ch   #256
        ch = 256  # h // 2      #256
        cw = 512  # square ch   #256

        h0 = random.randint(0, h - ch)  # 128
        w0 = random.randint(0, w - cw)  # 128
    else:
        ch, cw, h0, w0 = h, w, 0, 0

    if args.model == 'CvT_VS':
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


# 测试程序
def project_test(test_cover_dir, test_stego_dir, net):
    # 加载数据集
    test_loader = build_val_loader(
        test_cover_dir, test_stego_dir, batch_size=args.batch_size,
        num_workers=args.num_workers)
    import time
    test_time = time.time()
    net.eval()
    test_loss = 0.
    test_accuracy = 0.
    prgtime = 0

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = preprocess_data(data['image'], data['label'], True)

            if args.model == 'CvT_VS':  #
                start = time.time()
                outputs, feats_0, feats_1 = net(*inputs)
                prgtime += time.time() - start
                test_loss += criterion_1(outputs, labels).item() + 0.1 * criterion_2(feats_0, feats_1, labels)

            else:
                outputs = net(*inputs)
                test_loss += criterion_1(outputs, labels).item()

            test_accuracy += src.models.accuracy(outputs, labels).item()
    test_loss /= len(test_loader)
    test_accuracy /= len(test_loader)

    logger.info('Test set: Loss: {:.6f}, Accuracy: {:.2f}%'.format(
        test_loss, 100 * test_accuracy))
    logger.info('Test Time: {:.6f}s'.format(time.time() - test_time))
    return prgtime


if __name__ == '__main__':
    test_data_dir = "D:\shiyao_DataSet\Dataset\Alaska\EXP_DATA\\testVSize"
    VSize_dir = os.listdir(test_data_dir)

    # 加载最佳模型
    model_best_path = "D:\shiyao_DataSet\复现其他两个网络\exp\Vsize\CvT_VS"
    best_ckpt = os.path.join(model_best_path, 'model_best.pth.tar')  # epoch115checkpoint
    print("model_path:", best_ckpt)
    net.load_state_dict(torch.load(best_ckpt)['state_dict'])
    all_time = 0
    for i_filename in VSize_dir:
        test_cover_dir = os.path.join(test_data_dir, i_filename, '0')
        test_stego_dir = os.path.join(test_data_dir, i_filename, '1')
        logger.info('Final Test=====================================================')
        logger.info('VSize Name{}----------------------------------------------'.format(i_filename))
        time = project_test(test_cover_dir, test_stego_dir, net)
        all_time += time

    print('ALL test time: %.3f' % all_time)



