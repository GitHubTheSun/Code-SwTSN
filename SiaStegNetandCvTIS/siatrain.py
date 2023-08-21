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

Detection_target = 'sun0.1'
Name_model = 'sia'


def parse_args():
    parser = argparse.ArgumentParser()
    dataset_dir = "D:\shiyao_DataSet\Dataset\BOSSBase_256_bilinear\EXP/"
    ckpt_dir = 'D:\shiyao_DataSet\复现其他两个网络\exp/'
    train_cover_dir = os.path.join(dataset_dir, 'train', Detection_target, "0")
    train_stego_dir = os.path.join(dataset_dir, 'train', Detection_target, "1")
    valid_cover_dir = os.path.join(dataset_dir, 'valid', Detection_target, "0")
    valid_stego_dir = os.path.join(dataset_dir, 'valid', Detection_target, "1")
    ttest_cover_dir = os.path.join(dataset_dir, 'test', Detection_target, "0")
    ttest_stego_dir = os.path.join(dataset_dir, 'test', Detection_target, "1")
    ckpt_dir = os.path.join(ckpt_dir, Name_model, Detection_target)

    parser.add_argument('--train-cover-dir', dest='train_cover_dir', type=str, default=train_cover_dir)
    parser.add_argument('--train-stego-dir', dest='train_stego_dir', type=str, default=train_stego_dir)
    parser.add_argument('--val-cover-dir', dest='val_cover_dir', type=str, default=valid_cover_dir)
    parser.add_argument('--val-stego-dir', dest='val_stego_dir', type=str, default=valid_stego_dir)
    parser.add_argument('--test-cover-dir', dest='test_cover_dir', type=str, default=ttest_cover_dir)
    parser.add_argument('--test-stego-dir', dest='test_stego_dir', type=str, default=ttest_stego_dir)
    parser.add_argument('--ckpt-dir', dest='ckpt_dir', type=str, default=ckpt_dir)

    parser.add_argument('--epoch', dest='epoch', type=int, default=500)  # default=1000
    parser.add_argument('--random-crop', dest='random_crop', action='store')  # action='store_true')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=32)
    parser.add_argument('--num-workers', dest='num_workers', type=int, default=0)
    parser.add_argument('--model', dest='model', type=str, default=Name_model)
    parser.add_argument('--seed', dest='seed', type=int, default=-1)
    parser.add_argument('--log-interval', dest='log_interval', type=int, default=50)

    args = parser.parse_args()
    return args


def setup(args):
    os.makedirs(args.ckpt_dir, exist_ok=False)
    log_file = osp.join(args.ckpt_dir, 'log.txt')
    utils.configure_logging(file=log_file, root_handler_type=1)
    utils.set_random_seed(None if args.seed < 0 else args.seed)
    logger.info('Command Line Arguments: {}'.format(str(args)))


args = parse_args()
setup(args)
with open(os.path.join(args.ckpt_dir, 'results.csv'), 'w') as f:
    f.write('epoch,train_loss,train_acc,valid_loss,valid_acc,test_acc\n')

logger.info('Building data loader')
train_loader, epoch_length = build_train_loader(
    args.train_cover_dir, args.train_stego_dir, batch_size=args.batch_size,
    num_workers=args.num_workers
)
val_loader = build_val_loader(
    args.val_cover_dir, args.val_stego_dir, batch_size=args.batch_size,
    num_workers=args.num_workers
)
train_loader_iter = iter(train_loader)

logger.info('Building model')
if args.model == 'sia':
    net = src.models.KeNet().cuda()
    optimizer = Adamax(net.parameters(), lr=0.001, eps=1e-8, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300, 400], gamma=0.1)
elif args.model == 'cvt':
    net = src.models.CvT().cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300], gamma=0.1)
else:
    raise NotImplementedError

# 在低嵌入率下使用迁移学习
print("使用迁移学习：预加载在0.2嵌入率下的模型参数！")
net_parameters_path = "D:\shiyao_DataSet\复现其他两个网络\exp\sia\sun0.2"
best_ckpt = os.path.join(net_parameters_path, 'model_best.pth.tar')
net.load_state_dict(torch.load(best_ckpt)['state_dict'])

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

        h0 = random.randint(0, h - ch)  # 128
        w0 = random.randint(0, w - cw)  # 128
    else:
        ch, cw, h0, w0 = h, w, 0, 0

    if args.model == 'sia':
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


def train(epoch):
    net.train()
    running_loss, running_accuracy = 0., 0.

    for batch_idx in range(epoch_length):
        data = next(train_loader_iter)
        inputs, labels = preprocess_data(data['image'], data['label'], args.random_crop)

        # optimizer.zero_grad()
        if args.model == 'sia':  #
            outputs, feats_0, feats_1 = net(*inputs)
            loss = criterion_1(outputs, labels) + 0.1 * criterion_2(feats_0, feats_1, labels)

        else:
            outputs = net(*inputs)
            loss = criterion_1(outputs, labels)

        accuracy = src.models.accuracy(outputs, labels).item()
        running_accuracy += accuracy
        running_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % args.log_interval == 0:
            running_accuracy /= args.log_interval
            running_loss /= args.log_interval

            logger.info(
                'Train epoch: {} [{:3d}/{}]\tAccuracy: {:.2f}%\tLoss: {:.6f}'.format(
                    epoch, batch_idx + 1, epoch_length, 100 * running_accuracy,
                    running_loss))

            # ##############################log per log_interval start
            is_best = False
            save_checkpoint(
                {
                    'iteration': batch_idx + 1,
                    'state_dict': net.state_dict(),
                    'best_prec1': running_accuracy,
                    'optimizer': optimizer.state_dict(),
                },
                is_best,
                filename=os.path.join(args.ckpt_dir, 'checkpoint.pth.tar'),
                best_name=os.path.join(args.ckpt_dir, 'model_best.pth.tar'))
    return running_loss, running_accuracy


def valid():
    net.eval()
    valid_loss = 0.
    valid_accuracy = 0.
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = preprocess_data(data['image'], data['label'], False)

            if args.model == 'sia':  #
                outputs, feats_0, feats_1 = net(*inputs)
                valid_loss += criterion_1(outputs, labels).item() + 0.1 * criterion_2(feats_0, feats_1, labels)
            else:
                outputs = net(*inputs)
                valid_loss += criterion_1(outputs, labels).item()
            valid_accuracy += src.models.accuracy(outputs, labels).item()
    valid_loss /= len(val_loader)
    valid_accuracy /= len(val_loader)
    logger.info('Valid set: Loss: {:.6f}, Accuracy: {:.2f}%'.format(
        valid_loss, 100 * valid_accuracy))
    return valid_loss, valid_accuracy


def save_checkpoint(state, is_best, filename, best_name):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_name)


# 训练程序
best_accuracy = 0.
for e in range(1, args.epoch + 1):
    logger.info('Epoch: {}'.format(e))
    logger.info('Train----------------------------------------------------------')
    train_time = time.time()
    train_loss, train_accuracy = train(e)
    scheduler.step()
    logger.info('Train Time: {:.6f}s'.format(time.time() - train_time))
    logger.info('Valid----------------------------------------------------------')
    valid_time = time.time()
    valid_loss, valid_accuracy = valid()

    if valid_accuracy > best_accuracy:
        best_accuracy = valid_accuracy
        is_best = True
    else:
        is_best = False

    with open(os.path.join(args.ckpt_dir, 'results.csv'), 'a') as f:
        f.write('%03d,%0.6f,%0.6f,%0.5f,%0.5f,\n' % (
            e,
            train_loss,
            train_accuracy,
            valid_loss,
            valid_accuracy,
        ))
    logger.info('Best Accuracy: {:.2f}%'.format(100 * best_accuracy))
    logger.info('Valid Time: {:.6f}s'.format(time.time() - valid_time))

    save_checkpoint(
        {
            'epoch': e,
            'state_dict': net.state_dict(),
            'best_prec1': valid_accuracy,
            'optimizer': optimizer.state_dict(),
        },
        is_best,
        filename=os.path.join(args.ckpt_dir, 'checkpoint.pth.tar'),
        best_name=os.path.join(args.ckpt_dir, 'model_best.pth.tar'))


# 测试程序
def project_test():
    logger.info('Final Test=====================================================')
    # 加载数据集
    test_loader = build_val_loader(
        args.test_cover_dir, args.test_stego_dir, batch_size=args.batch_size,
        num_workers=args.num_workers)
    # 加载最佳模型
    best_ckpt = os.path.join(args.ckpt_dir, 'model_best.pth.tar')
    net.load_state_dict(torch.load(best_ckpt)['state_dict'])

    test_time = time.time()
    net.eval()
    test_loss = 0.
    test_accuracy = 0.
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = preprocess_data(data['image'], data['label'], False)
            if args.model == 'sia':  #
                outputs, feats_0, feats_1 = net(*inputs)
                test_loss += criterion_1(outputs, labels).item() + 0.1 * criterion_2(feats_0, feats_1, labels)
            else:
                outputs = net(*inputs)
                test_loss += criterion_1(outputs, labels).item()
            test_accuracy += src.models.accuracy(outputs, labels).item()
    test_loss /= len(test_loader)
    test_accuracy /= len(test_loader)

    with open(os.path.join(args.ckpt_dir, 'results.csv'), 'a') as f:
        f.write(',,,,,%0.5f\n' % test_accuracy)
    logger.info('Final Test set: Loss: {:.6f}, Accuracy: {:.2f}%'.format(
        test_loss, 100 * test_accuracy))
    logger.info('Final Test Time: {:.6f}s'.format(time.time() - test_time))


project_test()

