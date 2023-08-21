
from tool import *
from tool import _label_sum
import os
import datetime
import torch
import torchvision
import torch.nn.functional as F
from torch.backends import cudnn
from torch.utils.data.dataset import Dataset
from torchvision.datasets import ImageFolder
from torch.autograd import Variable

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def _label_sum(pred, target):
    pred = pred.view_as(target)
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    l1 = []
    for i in range(len(target)):
        l1.append(pred[i] + target[i])
    # l1.count(0)即为 正确被判定为载体图像（阴性）的数量。l1.count(2)，即为正确被判定为载密图像（阳性）的数量。l1.count(0)+l1.count(2) 即为判断正确的总个数
    return l1.count(0), l1.count(2), l1.count(0) + l1.count(2)


def test_epoch(model, loader, is_test=True):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    prgtime = 0

    # Model on eval mode
    model.eval()
    import time
    # end = time.time()
    with torch.no_grad():
        for batch_idx, datas in enumerate(loader):
            # Create vaiables
            images, labels = datas

            images = Variable(images.to(device))
            labels = Variable(labels.to(device))

            # compute output
            start = time.time()
            outputs = model(images)
            prgtime += time.time() - start
            # loss = torch.nn.functional.cross_entropy(outputs, labels)
            # _, pred = outputs.data.cpu().topk(1, dim=1)

            output1 = F.log_softmax(outputs, dim=1)
            loss = F.nll_loss(output1, labels)
            pred = outputs.max(1, keepdim=True)[1]

            # measure accuracy and record loss
            batch_size = labels.size(0)
            # _, pred = outputs.data.cpu().topk(1, dim=1)
            # acc.update(torch.ne(pred.squeeze(), labels.cpu()).float().sum().item() / batch_size, batch_size)
            acc.update(pred.eq(labels.view_as(pred)).sum().item() / batch_size, batch_size)
            losses.update(loss.item() / batch_size, batch_size)

            # batch_time.update(time.time() - end)
            # end = time.time()

        res = '\t'.join([
            'Test:' if is_test else 'Valid:',
            'Time %.3f s' % batch_time.sum,
            'Loss %.4f ' % losses.avg,
            'Acc %.4f ' % acc.avg,
        ])
        print(res)

    # Return summary statistics
    # return batch_time.avg, losses.avg, acc.avg, prgtime
    return prgtime


if __name__ == '__main__':
    test_data_dir = "D:\shiyao_DataSet\Dataset\Alaska\EXP_DATA\\testVSize"
    VSize_dir = os.listdir(test_data_dir)

    data_transform = torchvision.transforms.Compose([torchvision.transforms.Grayscale(),
                                                     torchvision.transforms.ToTensor()])

    # 加载最佳模型
    from SwTDSN import SwTDSN
    model = SwTDSN()
    num_params = sum(p.numel() for p in model.parameters())
    print("Total parameters: ", num_params)

    model_best_path = "C:\\Users\Lenovo\Desktop\\TWODAYSEXPRE\\New_FE_8_2_NPE_VSize"
    best_ckpt = os.path.join(model_best_path, 'best_model.dat')
    print("model_Path:", best_ckpt)
    model.load_state_dict(torch.load(best_ckpt))
    model = model.cuda()

    all_time = 0

    for i_filename in VSize_dir:
        cover_dir = os.path.join(test_data_dir, i_filename)

        Testfilepath = cover_dir + '\\'
        test_set = torch.utils.data.DataLoader(
            ImageFolder(Testfilepath, data_transform),
            batch_size=8 * 2,
            shuffle=False)

        print('Final Test=====================================================')
        print('VSize Name{}----------------------------------------------'.format(i_filename))
        test_results = test_epoch(
            model=model,
            loader=test_set,
            is_test=True
        )
        time = test_results
        all_time += time
        print('Final test time: %.3f' % time)
    print('ALL test time: %.3f' % all_time)