import numpy as np
import torch
import torch.nn as nn

ddac_npy = np.load('D:\shiyao_DataSet\Cvt-steganalysis\project\other/ddac_kernels.npy')
ddac_npy = torch.from_numpy(ddac_npy)
ddac_npy = ddac_npy.type(torch.FloatTensor).cuda()

# import matplotlib.pyplot as plt
# plt.figure()
# plt.imshow(ddac_npy.view(8, 3, 3)[2], cmap="Greys_r")
# plt.figure()
# plt.imshow(ddac_npy.view(8, 3, 3)[3], cmap="Greys_r")
# plt.figure()
# plt.imshow(ddac_npy.view(8, 3, 3)[4], cmap="Greys_r")
# plt.figure()
# plt.imshow(ddac_npy.view(8, 3, 3)[5], cmap="Greys_r")
# plt.figure()
# plt.imshow(ddac_npy.view(8, 3, 3)[6], cmap="Greys_r")
# plt.figure()
# plt.imshow(ddac_npy.view(8, 3, 3)[7], cmap="Greys_r")
# plt.show()

SRM_npy = np.load('D:\shiyao_DataSet\Cvt-steganalysis\project\preprocessing\SRM_Kernels.npy')
# print(SRM_npy)
SRM_npy = torch.tensor(SRM_npy).cuda()


class DDAC(nn.Module):
    def __init__(self):
        super(DDAC, self).__init__()
        self.ddac = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3),
                              padding=1, stride=(1, 1))
        self.conv = nn.Conv2d(in_channels=8, out_channels=30, kernel_size=(1, 1),
                              padding=0, stride=(1, 1))

        self.TLU = nn.Hardtanh(-10, 10, True)

        self.ddac.weight = torch.nn.Parameter(ddac_npy, requires_grad=True)

    def forward(self, x):
        x = self.ddac(x)
        x = self.conv(x)
        x = self.TLU(x)
        return x


class SRM(nn.Module):
    def __init__(self):
        super(SRM, self).__init__()
        self.Conv_SRM = nn.Conv2d(in_channels=1, out_channels=30, kernel_size=(5, 5),
                                  padding=2, stride=(1, 1))
        self.Conv_SRM_bn = nn.BatchNorm2d(30)

        self.Conv_SRM.weight = torch.nn.Parameter(SRM_npy, requires_grad=False)

    def forward(self, x):
        x = self.Conv_SRM(x)
        x = self.Conv_SRM_bn(x)
        return x
