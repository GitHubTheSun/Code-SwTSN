from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter
from einops import repeat
from einops.layers.torch import Rearrange
import numpy as np
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch
from torch import nn, einsum
from einops import rearrange


def accuracy(outputs, labels):
    _, argmax = torch.max(outputs, 1)
    return (labels == argmax.squeeze()).float().mean()


SRM_npy = np.load('D:\shiyao_DataSet\复现其他两个网络\SiaStegNet_master\SiaStegNet_master\src\models/SRM_Kernels.npy')


class SepConv2d(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1, ):
        super(SepConv2d, self).__init__()
        self.depthwise = torch.nn.Conv2d(in_channels,
                                         in_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=in_channels)
        self.bn = torch.nn.BatchNorm2d(in_channels)
        self.pointwise = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class ConvAttention(nn.Module):
    def __init__(self, dim, img_size_h, img_size_w, heads=8, dim_head=64, kernel_size=3, q_stride=1, k_stride=1,
                 v_stride=1, dropout=0.,
                 last_stage=False):

        super().__init__()
        self.last_stage = last_stage
        self.img_size_h = img_size_h
        self.img_size_w = img_size_w
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        pad = (kernel_size - q_stride) // 2
        self.to_q = SepConv2d(dim, inner_dim, kernel_size, q_stride, pad)
        self.to_k = SepConv2d(dim, inner_dim, kernel_size, k_stride, pad)
        self.to_v = SepConv2d(dim, inner_dim, kernel_size, v_stride, pad)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        if self.last_stage:
            cls_token = x[:, 0]
            x = x[:, 1:]
            # print('pre_cls_token.shape',cls_token.shape)
            cls_token = rearrange(cls_token.unsqueeze(1), 'b n (h d) -> b h n d', h=h)
            # print('now_cls_token.shape',cls_token.shape)
        x = rearrange(x, 'b (l w) n -> b n l w', l=self.img_size_h, w=self.img_size_w)
        q = self.to_q(x)
        q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=h)

        v = self.to_v(x)
        v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=h)

        k = self.to_k(x)
        k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=h)

        if self.last_stage:
            # print('cls_token.shape',cls_token.shape)
            # print('q.shape', q.shape)
            # print('k.shape', q.shape)
            # print('v.shape', q.shape)
            q = torch.cat((cls_token, q), dim=2)
            v = torch.cat((cls_token, v), dim=2)
            k = torch.cat((cls_token, k), dim=2)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class SRM_conv2d(nn.Module):
    def __init__(self, stride=1, padding=0):
        super(SRM_conv2d, self).__init__()
        self.in_channels = 1
        self.out_channels = 30
        self.kernel_size = (5, 5)
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding
        self.dilation = (1, 1)
        self.transpose = False
        self.groups = 1
        self.weight = Parameter(torch.Tensor(30, 1, 5, 5), \
                                requires_grad=True)
        self.bias = Parameter(torch.Tensor(30), \
                              requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.numpy()[:] = SRM_npy
        self.bias.data.zero_()

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, \
                        self.stride, self.padding, self.dilation, \
                        self.groups)


class Transformer(nn.Module):
    def __init__(self, dim, img_size_h, img_size_w, depth, heads, dim_head, mlp_dim, dropout=0., last_stage=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, ConvAttention(dim, img_size_h, img_size_w, heads=heads, dim_head=dim_head, dropout=dropout,
                                           last_stage=last_stage)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            # print('x.shape',x.shape)
            x = attn(x) + x
            x = ff(x) + x
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, \
                 stride=1, with_bn=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, \
                              stride)
        self.bn = nn.BatchNorm2d(30)
        self.relu = nn.ReLU()
        self.with_bn = with_bn
        if with_bn:
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = lambda x: x
        self.reset_parameters()

    def forward(self, x):
        return self.norm(self.relu(self.bn(self.conv(x))))

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.conv.weight)
        self.conv.bias.data.fill_(0.2)
        if self.with_bn:
            self.norm.reset_parameters()


class SELayer(nn.Module):
    def __init__(self, channel):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, 6, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(6, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class BlockA(nn.Module):

    def __init__(self, in_planes, out_planes, norm_layer=None):
        super(BlockA, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = conv3x3(in_planes, out_planes)
        self.bn1 = norm_layer(out_planes)
        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2 = norm_layer(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class BlockC(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super(BlockC, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        out += residual
        out = self.relu(out)

        return out


class YeNet(nn.Module):
    def __init__(self, with_bn=False, norm_layer=None, threshold=3):
        super(YeNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.with_bn = with_bn
        self.preprocessing = SRM_conv2d(1, 2)
        self.bn1 = norm_layer(30)
        self.relu = nn.ReLU(inplace=True)

        self.AA = BlockC(30, 30)
        self.A1 = BlockA(30, 30, norm_layer=norm_layer)
        self.A2 = BlockA(30, 30, norm_layer=norm_layer)

    def forward(self, x):
        x = x.float()
        x = self.preprocessing(x)

        x = self.bn1(x)
        x = self.relu(x)
        x = self.AA(x)
        x = self.A1(x)
        x = self.A2(x)

        return x


class CvT_pre(nn.Module):
    def __init__(self, image_size_h=256, image_size_w=256, in_channels=30, num_classes=2, dim=16,
                 kernels=[3, 3, 1, 3, 3], strides=[2, 2, 2, 2, 2],
                 heads=[1, 4, 8, 16, 32], depth=[1, 2, 2, 2, 2], pool='cls', dropout=0., emb_dropout=0., scale_dim=4):
        super().__init__()

        self.yenet = YeNet()

        # 三层嵌入操作
        self.layerStage1 = nn.Conv2d(in_channels=30, out_channels=16,
                                     kernel_size=kernels[0], stride=strides[0], padding=1, bias=False)
        self.bnStage1 = nn.BatchNorm2d(16)

        self.layerStage2 = nn.Conv2d(in_channels=16, out_channels=16,
                                     kernel_size=kernels[0], stride=strides[0], padding=1, bias=False)
        self.bnStage2 = nn.BatchNorm2d(16)

        self.layerStage3 = nn.Conv2d(in_channels=16, out_channels=16,
                                     kernel_size=kernels[0], stride=1, padding=1, bias=False)
        self.bnStage3 = nn.BatchNorm2d(16)

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pool = pool
        self.dim = dim

        # #### Stage 1 #######
        self.stage1_conv_embed = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernels[0], strides[0], 1),
            nn.Conv2d(dim, dim, kernels[0], strides[0], 1),
            nn.Conv2d(dim, dim, kernels[0], 1, 1),
            Rearrange('b c h w -> b (h w) c', h=image_size_h // 4, w=image_size_w // 4),
            nn.LayerNorm(dim)
        )
        self.stage1_transformer = nn.Sequential(
            Transformer(dim=dim, img_size_h=image_size_h // 4, img_size_w=image_size_w // 4, depth=depth[0],
                        heads=heads[0], dim_head=self.dim,
                        mlp_dim=dim * scale_dim, dropout=dropout),
            Rearrange('b (h w) c -> b c h w', h=image_size_h // 4, w=image_size_w // 4)
        )

        # #### Stage 2 #######
        in_channels = dim
        scale = heads[1] // heads[0]
        dim = scale * dim
        self.stage2_conv_embed = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernels[1], strides[1], 1),
            Rearrange('b c h w -> b (h w) c', h=image_size_h // 8, w=image_size_w // 8),
            nn.LayerNorm(dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, (image_size_h // 8) ** 2 + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.stage2_transformer = Transformer(dim=dim, img_size_h=image_size_h // 8, img_size_w=image_size_w // 8, depth=depth[1],
                                              heads=heads[1], dim_head=self.dim,
                                              mlp_dim=dim * scale_dim, dropout=dropout, last_stage=True)

        self.dropout_large = nn.Dropout(emb_dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        # yenet中是预处理部分 包含了srm结构
        pre_res = self.yenet(img)  # 'b,c,h,w',1,30,256,256
        # print('pre_res.shape', pre_res.shape)

        # 短接部分的卷积操作
        convs = self.layerStage1(pre_res)  # 'b,c,h,w',1,16,128,128
        convs = self.bnStage1(convs)
        # print('convs1.shape', convs.shape)

        convs = self.layerStage2(convs)  # 'b,c,h,w',1,16,64,64
        convs = self.bnStage2(convs)
        # print('convs2.shape', convs.shape)

        convs = self.layerStage3(convs)  # 'b,c,h,w',1,16,64,64
        convs = self.bnStage3(convs)
        # print('convs3.shape', convs.shape)

        # 主路上的嵌入部分
        xs = self.stage1_conv_embed(pre_res)
        # print('stage1.xs.conv.shape', xs.shape)
        xs = self.stage1_transformer(xs)
        # print('stage1.xs.trans.shape',xs.shape)

        xs = torch.add(convs, xs)
        # print('stage2.xs.shape', xs.shape)
        xs = self.stage2_conv_embed(xs)
        # print('stage2.xs.conv.shape', xs.shape)
        b, n, _ = xs.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        # print('cls_tokens.shape',cls_tokens.shape)
        xs = torch.cat((cls_tokens, xs), dim=1)
        xs += self.pos_embedding[:, :(n + 1)]
        xs = self.dropout_large(xs)
        # print('xs.new.shape',xs.shape)

        xs = self.stage2_transformer(xs)

        xs = xs.mean(dim=1) if self.pool == 'mean' else xs[:, 0]
        # xs = xs.mean(dim=1)
        # xs = xs.max(dim=1)
        # print(xs.shape)
        xs = torch.squeeze(xs)
        xs = self.mlp_head(xs)
        # print('xs.shape', print(xs.shape))
        return xs


class CvT(nn.Module):
    def __init__(self, image_size_h=256, image_size_w=256, in_channels=30, num_classes=2, dim=16,
                 kernels=[3, 3, 1, 3, 3], strides=[2, 2, 2, 2, 2],
                 heads=[1, 4, 8, 16, 32], depth=[1, 2, 2, 2, 2], pool='cls', dropout=0., emb_dropout=0., scale_dim=4):
        super().__init__()

        self.yenet = YeNet()

        # 三层嵌入操作
        self.layerStage1 = nn.Conv2d(in_channels=30, out_channels=16,
                                     kernel_size=kernels[0], stride=strides[0], padding=1, bias=False)
        self.bnStage1 = nn.BatchNorm2d(16)

        self.layerStage2 = nn.Conv2d(in_channels=16, out_channels=16,
                                     kernel_size=kernels[0], stride=strides[0], padding=1, bias=False)
        self.bnStage2 = nn.BatchNorm2d(16)

        self.layerStage3 = nn.Conv2d(in_channels=16, out_channels=16,
                                     kernel_size=kernels[0], stride=1, padding=1, bias=False)
        self.bnStage3 = nn.BatchNorm2d(16)

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pool = pool
        self.dim = dim

        # #### Stage 1 #######
        self.stage1_conv_embed = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernels[0], strides[0], 1),
            nn.Conv2d(dim, dim, kernels[0], strides[0], 1),
            nn.Conv2d(dim, dim, kernels[0], 1, 1),
            Rearrange('b c h w -> b (h w) c', h=image_size_h // 4, w=image_size_w // 4),
            nn.LayerNorm(dim)
        )
        self.stage1_transformer = nn.Sequential(
            Transformer(dim=dim, img_size_h=image_size_h // 4, img_size_w=image_size_w // 4, depth=depth[0],
                        heads=heads[0], dim_head=self.dim,
                        mlp_dim=dim * scale_dim, dropout=dropout),
            Rearrange('b (h w) c -> b c h w', h=image_size_h // 4, w=image_size_w // 4)
        )

        # #### Stage 2 #######
        in_channels = dim
        scale = heads[1] // heads[0]
        dim = scale * dim
        self.stage2_conv_embed = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernels[1], strides[1], 1),
            Rearrange('b c h w -> b (h w) c', h=image_size_h // 8, w=image_size_w // 8),
            nn.LayerNorm(dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, (image_size_h // 8) ** 2 + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.stage2_transformer = Transformer(dim=dim, img_size_h=image_size_h // 8, img_size_w=image_size_w // 8, depth=depth[1],
                                              heads=heads[1], dim_head=self.dim,
                                              mlp_dim=dim * scale_dim, dropout=dropout, last_stage=True)

        self.dropout_large = nn.Dropout(emb_dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        # yenet中是预处理部分 包含了srm结构
        pre_res = self.yenet(img)  # 'b,c,h,w',1,30,256,256
        # print('pre_res.shape', pre_res.shape)

        # 短接部分的卷积操作
        convs = self.layerStage1(pre_res)  # 'b,c,h,w',1,16,128,128
        convs = self.bnStage1(convs)
        # print('convs1.shape', convs.shape)

        convs = self.layerStage2(convs)  # 'b,c,h,w',1,16,64,64
        convs = self.bnStage2(convs)
        # print('convs2.shape', convs.shape)

        convs = self.layerStage3(convs)  # 'b,c,h,w',1,16,64,64
        convs = self.bnStage3(convs)
        # print('convs3.shape', convs.shape)

        # 主路上的嵌入部分
        xs = self.stage1_conv_embed(pre_res)
        # print('stage1.xs.conv.shape', xs.shape)
        xs = self.stage1_transformer(xs)
        # print('stage1.xs.trans.shape',xs.shape)

        xs = torch.add(convs, xs)
        # print('stage2.xs.shape', xs.shape)
        xs = self.stage2_conv_embed(xs)
        # print('stage2.xs.conv.shape', xs.shape)
        b, n, _ = xs.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        # print('cls_tokens.shape',cls_tokens.shape)
        xs = torch.cat((cls_tokens, xs), dim=1)
        xs += self.pos_embedding[:, :(n + 1)]
        xs = self.dropout_large(xs)
        # print('xs.new.shape',xs.shape)

        xs = self.stage2_transformer(xs)

        xs = xs.mean(dim=1) if self.pool == 'mean' else xs[:, 0]
        # xs = xs.mean(dim=1)
        # xs = xs.max(dim=1)
        # print(xs.shape)
        # xs = torch.squeeze(xs)
        # xs = self.mlp_head(xs)
        # print('xs.shape', print(xs.shape))
        return xs


def load_model_parameters(net, model_path):
    model_parameters = torch.load(model_path)
    net.load_state_dict(model_parameters)
    return net


def changeModelParam(model1, model2):  # model2是被替换者
    for name2, param2 in model2.named_parameters():
        for name1, param1 in model1.named_parameters():
            if name2 == name1:
                param2 = param1
                print(name2)
    print("model1和model2调换参数完成！")
    return model2


class CvT_VS(nn.Module):

    def __init__(self, norm_layer=None, zero_init_residual=True, p=0.5):
        super(CvT_VS, self).__init__()

        self.cvt = CvT()
        CvT_pre_net = CvT_pre().cuda()
        CvT_pre_model_path = "D:\shiyao_DataSet\复现其他两个网络\exp\Vsize\cvtvswow0.4pre10epoch\\checkpoint.pth.tar"
        CvT_pre_net.load_state_dict(torch.load(CvT_pre_model_path)['state_dict'])
        self.cvt = changeModelParam(CvT_pre_net, self.cvt)

        self.fc = nn.Linear(64 * 4 + 1, 2)
        self.dropout = nn.Dropout(p=p)

    def extract_feat(self, x):
        x = x.float()
        out = self.cvt(x)
        # print(out.shape)
        out = out.view(out.size(0), out.size(1))
        return out

    def forward(self, *args):
        ############# statistics fusion start #############
        feats = torch.stack(
            [self.extract_feat(subarea) for subarea in args], dim=0
        )

        euclidean_distance = F.pairwise_distance(feats[0], feats[1], eps=1e-6, keepdim=True)

        if feats.shape[0] == 1:
            final_feat = feats.squeeze(dim=0)
        else:
            feats_mean = feats.mean(dim=0)
            feats_var = feats.var(dim=0)
            feats_min, _ = feats.min(dim=0)
            feats_max, _ = feats.max(dim=0)

            final_feat = torch.cat(
                [euclidean_distance, feats_mean, feats_var, feats_min, feats_max], dim=-1
            )

        out = self.dropout(final_feat)
        out = self.fc(out)

        return out, feats[0], feats[1]


if __name__ == "__main__":
    img = torch.randn(16, 1, 256, 256)

    model = CvT_VS().cuda()

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)

    img = img.cuda()
    out = model(img).cuda()
    print("Shape of out :", out.shape)

    # img = img.cuda()
    # out = model(img).cuda()
    #
    # print("Shape of out :", out.shape)  # [B, num_classes]

    '''
    # 测试不同尺寸下的方式
    size = [512, 640, 720, 1024]
    for h in size:
        for w in size:
            img = torch.randn(16, 1, h, w)
            img = img.cuda()
            out = model(img).cuda()
            print("Shape of out :", out.shape)
    '''
