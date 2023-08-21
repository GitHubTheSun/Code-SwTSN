import torch.nn as nn
import torch
import math


class CSPP_SID(nn.Module):
    def __init__(self, patch_size=8, input_dim=30, output_dim=64):
        super(CSPP_SID, self).__init__()
        self.patch_size = patch_size
        self.output_dim = output_dim
        self.stage2_conv = nn.Sequential(nn.Conv2d(1, 4, kernel_size=7, stride=3, padding=0),
                                         # if patch size is 64 kernel_size=3, stride=1
                                         # nn.BatchNorm2d(4),
                                         )
        self.stage3_conv = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=0),
                                         # if patch size is 64 kernel_size=2, stride=2
                                         # nn.BatchNorm2d(16),
                                         )
        self.fclayer = nn.Linear(84 * input_dim, output_dim)

    def spp_sub(self, x, mode):
        x = torch.unsqueeze(x, 3)
        b, c, n, spp_c, h, w = x.shape
        x = x.reshape(b * c * n, spp_c, h, w)
        if mode == 1:
            spp_c = 1
        elif mode == 2:
            x = self.stage2_conv(x)
            spp_c = 4
        elif mode == 3:
            x = self.stage3_conv(x)
            spp_c = 16
        x = x.reshape(b, c, n, spp_c, -1)
        x_min = torch.min(x, dim=4).values
        x_max = torch.max(x, dim=4).values
        x_avg = torch.mean(x, dim=4)
        x_var = torch.var(x, dim=4)
        output = torch.cat([torch.unsqueeze(x_max, 4),
                            torch.unsqueeze(x_min, 4),
                            torch.unsqueeze(x_avg, 4),
                            torch.unsqueeze(x_var, 4)], dim=4)
        return output

    def spp(self, x):
        stage1_mode = 1
        stage2_mode = 2
        stage3_mode = 3
        x1 = self.spp_sub(x, stage1_mode)
        x2 = self.spp_sub(x, stage2_mode)
        x3 = self.spp_sub(x, stage3_mode)
        output = torch.cat([x1, x2, x3], dim=3)
        return output

    def forward(self, input):
        b, c, h, w = input.shape
        assert (h // self.patch_size or w // self.patch_size) is not True, \
            f"Input image size ({h}*{w}) doesn't match model ({h}*{w})."
        P_h, P_w = self.patch_size, self.patch_size
        N_h, N_w = h // self.patch_size, w // self.patch_size
        x = input.reshape(b, c, P_h, N_h, P_w, N_w).permute(0, 1, 2, 4, 3, 5)
        x = x.reshape(b, c, P_h * P_w, N_h, N_w)
        x = self.spp(x)
        x = x.permute(0, 2, 1, 3, 4).reshape(b, P_h * P_w, -1)
        output = self.fclayer(x)
        return output


class SPP(nn.Module):
    def __init__(self, patch_size=8, input_dim=30, output_dim=64):
        super(SPP, self).__init__()
        self.patch_size = patch_size
        self.output_dim = output_dim
        self.fclayer = nn.Linear(21 * input_dim, output_dim)  # 30*21 -> 64

    def method_SPP(self, x):  # [b, c, n, p1, p2] -> [b, c, n, 21]
        poolpatch_num = [1, 2, 4]
        b, c, n, h, w = x.shape
        x = x.reshape(b * c, n, h, w)
        for i in range(len(poolpatch_num)):
            h_wid = int(math.ceil(h / poolpatch_num[i]))
            w_wid = int(math.ceil(w / poolpatch_num[i]))
            h_pad = int((h_wid * poolpatch_num[i] - h + 1) / 2)
            w_pad = int((w_wid * poolpatch_num[i] - w + 1) / 2)
            maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
            sub_output = maxpool(x)
            if i == 0:
                output = sub_output.view(b * c, n, -1)
            else:
                sub_output = sub_output.view(b * c, n, -1)
                output = torch.cat((output, sub_output), dim=2)
        output = output.reshape(b, c, n, -1)
        return output

    def forward(self, input):
        b, c, h, w = input.shape
        assert (h // self.patch_size or w // self.patch_size) is not True, \
            f"Input image size ({h}*{w}) doesn't match model ({h}*{w})."
        P_h, P_w = self.patch_size, self.patch_size
        N_h, N_w = h // self.patch_size, w // self.patch_size
        x = input.reshape(b, c, P_h, N_h, P_w, N_w).permute(0, 1, 2, 4, 3, 5)
        x = x.reshape(b, c, P_h * P_w, N_h, N_w)
        x = self.method_SPP(x)  # [b, c, n, p1, p2] -> [b, c, n, 21]
        x = x.permute(0, 2, 1, 3).reshape(b, P_h * P_w, -1)
        output = self.fclayer(x)
        return output


class SID(nn.Module):
    def __init__(self, patch_size=8, input_dim=30, output_dim=64):
        super(SID, self).__init__()
        self.patch_size = patch_size
        self.output_dim = output_dim
        self.fclayer = nn.Linear(4 * input_dim, output_dim)  # 120 -> 64

    def method_SID(self, x):  # [b, c, n, p1, p2] -> [b, c, n, 4]
        b, c, n, p1, p2 = x.shape
        x = x.reshape(b, c, n, -1)
        x_min = torch.min(x, dim=3).values
        x_max = torch.max(x, dim=3).values
        x_avg = torch.mean(x, dim=3)
        x_var = torch.var(x, dim=3)
        output = torch.cat([torch.unsqueeze(x_max, 3),
                            torch.unsqueeze(x_min, 3),
                            torch.unsqueeze(x_avg, 3),
                            torch.unsqueeze(x_var, 3)], dim=3)
        return output  # [b, c, n, 4]

    def forward(self, input):
        b, c, h, w = input.shape
        assert (h // self.patch_size or w // self.patch_size) is not True, \
            f"Input image size ({h}*{w}) doesn't match model ({h}*{w})."
        P_h, P_w = self.patch_size, self.patch_size
        N_h, N_w = h // self.patch_size, w // self.patch_size
        x = input.reshape(b, c, P_h, N_h, P_w, N_w).permute(0, 1, 2, 4, 3, 5)
        x = x.reshape(b, c, P_h * P_w, N_h, N_w)
        x = self.method_SID(x)  # [b, c, n, p1, p2]
        x = x.permute(0, 2, 1, 3).reshape(b, P_h * P_w, -1)
        output = self.fclayer(x)
        return output