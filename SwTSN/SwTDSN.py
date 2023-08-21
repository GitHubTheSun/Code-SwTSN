
from SwinTransformer.SwTBlock import *
from timm.models.layers import trunc_normal_

import numpy as np
from SRM_DDAC.getSRM_DDAC import *
from net_otherstruct import *


class SELayer(nn.Module):
    def __init__(self, channel, reduction=5):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel, bias=False),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        # print(y, self.fc(y))
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Processing(nn.Module):
    def __init__(self):
        super(Processing, self).__init__()

        self.ddac = DDAC()
        channel = 30

        self.Resnet_50_sub1 = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(1, 1), padding=0, stride=(1, 1)),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(3, 3), padding=1, stride=(1, 1)),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
        )

        self.selayer = SELayer(channel)
        self.Resnet_50_sub2 = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(1, 1), padding=0, stride=(1, 1)),
            nn.BatchNorm2d(channel),
        )
        self.Resnet_50_relu = nn.ReLU(inplace=True)

        self.Resnet_18_1 = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(3, 3), padding=1, stride=(1, 1)),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(3, 3), padding=1, stride=(1, 1)),
            nn.BatchNorm2d(channel),
        )
        self.Resnet_18_1_relu = nn.ReLU(inplace=True)

        self.Resnet_18_2 = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(3, 3), padding=1, stride=(1, 1)),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(3, 3), padding=1, stride=(1, 1)),
            nn.BatchNorm2d(channel),
        )
        self.Resnet_18_2_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        ddacx = self.ddac(x)

        x = ddacx + self.Resnet_50_sub2(self.selayer(self.Resnet_50_sub1(ddacx)))
        x = self.Resnet_50_relu(x)

        x = x + self.Resnet_18_1(x)
        x = self.Resnet_18_1_relu(x)

        x = x + self.Resnet_18_2(x)
        x = self.Resnet_18_2_relu(x)
        return x


class SwTDSN(nn.Module):
    def __init__(self, num_classes=2, embed_dim=64, depths=[2], num_heads=[4],
                 window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, fused_window_process=False):
        super().__init__()

        self.processing = Processing()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))  # 结构尾部的norm
        self.mlp_ratio = mlp_ratio

        self.patch_embed = CSPP_SID(patch_size=8, input_dim=30)  # four
        num_patches = 8*8
        patches_resolution = [8, 8]
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               fused_window_process=fused_window_process)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.processing(x)
        # print(x.shape)
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


if __name__ == '__main__':
    import time
    import datetime
    a = torch.randn(1, 1, 256, 256)
    a = a.cuda()
    net = SwTDSN().cuda()
    print(net)
    num_params = sum(p.numel() for p in net.parameters())
    print("Total parameters: ", num_params)
    since = time.time()
    b = net(a)
    end = time.time()
    print(b.shape, end - since)
