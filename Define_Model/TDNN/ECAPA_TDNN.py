#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: ECAPA_TDNN.py
@Time: 2021/5/1 08:36
@Overview:
https://github.com/lawlict/ECAPA-TDNN/blob/master/ecapa_tdnn.py

"""
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from Define_Model.FilterLayer import Mean_Norm, TimeMaskLayer, FreqMaskLayer, TimeFreqMaskLayer
from Define_Model.model import get_filter_layer, get_mask_layer, get_input_norm

''' Res2Conv1d + BatchNorm1d + ReLU
'''


class Res2Conv1dReluBn(nn.Module):
    '''
    in_channels == out_channels == channels
    '''

    def __init__(self, channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, scale=4):
        super().__init__()
        assert channels % scale == 0, "{} % {} != 0".format(channels, scale)
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1

        self.convs = []
        self.bns = []
        for i in range(self.nums):
            self.convs.append(nn.Conv1d(self.width, self.width,
                              kernel_size, stride, padding, dilation, bias=bias))
            self.bns.append(nn.BatchNorm1d(self.width))
        self.convs = nn.ModuleList(self.convs)
        self.bns = nn.ModuleList(self.bns)

    def forward(self, x):
        out = []
        spx = torch.split(x, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            # Order: conv -> relu -> bn
            sp = self.convs[i](sp)
            sp = self.bns[i](F.relu(sp))
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        out = torch.cat(out, dim=1)
        return out


''' Conv1d + BatchNorm1d + ReLU
'''


class Conv1dReluBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size, stride, padding, dilation, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))


''' The SE connection of 1D case.
'''


class SE_Connect(nn.Module):
    def __init__(self, channels, s=2):
        super().__init__()
        assert channels % s == 0, "{} % {} != 0".format(channels, s)
        self.linear1 = nn.Linear(channels, channels // s)
        self.linear2 = nn.Linear(channels // s, channels)

    def forward(self, x):
        out = x.mean(dim=2)
        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        out = x * out.unsqueeze(2)
        return out


''' SE-Res2Block.
    Note: residual connection is implemented in the ECAPA_TDNN model, not here.
'''


def SE_Res2Block(channels, kernel_size, stride, padding, dilation, scale):
    return nn.Sequential(
        Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
        Res2Conv1dReluBn(channels, kernel_size, stride,
                         padding, dilation, scale=scale),
        Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
        SE_Connect(channels)
    )


''' Attentive weighted mean and standard deviation pooling.
'''


class AttentiveStatsPool(nn.Module):
    def __init__(self, in_dim, bottleneck_dim):
        super().__init__()
        # Use Conv1d with stride == 1 rather than Linear, then we don't need to transpose inputs.
        # equals W and b in the paper
        self.linear1 = nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1)
        # equals V and k in the paper
        self.linear2 = nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1)

    def forward(self, x):
        # DON'T use ReLU here! In experiments, I find ReLU hard to converge.
        alpha = torch.tanh(self.linear1(x))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        residuals = torch.sum(alpha * x ** 2, dim=2) - mean ** 2
        std = torch.sqrt(residuals.clamp(min=1e-9))
        return torch.cat([mean, std], dim=1)


''' Implementation of
    "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification".
    Note that we DON'T concatenate the last frame-wise layer with non-weighted mean and standard deviation, 
    because it brings little improvment but significantly increases model parameters. 
    As a result, this implementation basically equals the A.2 of Table 2 in the paper.
'''


# num_classes, embedding_size, input_dim, alpha=0., input_norm='',
#                  filter=None, sr=16000, feat_dim=64, exp=False, filter_fix=False,
#                  dropout_p=0.0, dropout_layer=False, encoder_type='STAP',
#                  num_classes_b=0, block_type='basic',
#                  mask='None', mask_len=20, channels=[512, 512, 512, 512, 1500], **kwargs
class ECAPA_TDNN(nn.Module):
    def __init__(self, num_classes, embedding_size=192, input_dim=80, input_norm='',
                 filter=None, sr=16000, feat_dim=80, exp=False, filter_fix=False,
                 dropout_p=0.0, dropout_layer=False, encoder_type='STAP',
                 num_classes_b=0, block_type='basic', alpha=0.,
                 init_weight='mel', scale=0.2, weight_p=0.1, weight_norm='max',
                 mask='None', mask_len=[5, 20], channels=[512, 512, 512, 512, 1536], **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_classes_b = num_classes_b
        self.dropout_p = dropout_p
        self.dropout_layer = dropout_layer
        self.input_dim = input_dim
        self.channels = channels
        self.alpha = alpha
        self.mask = mask
        self.filter = filter
        self.feat_dim = feat_dim
        self.block_type = block_type.lower()
        self.embedding_size = embedding_size

        input_mask = []
        filter_layer = get_filter_layer(filter=filter, input_dim=input_dim, sr=sr, feat_dim=feat_dim,
                                        exp=exp, filter_fix=filter_fix)
        if filter_layer != None:
            input_mask.append(filter_layer)

        norm_layer = get_input_norm(input_norm, input_dim=input_dim)
        if norm_layer != None:
            input_mask.append(norm_layer)
        mask_layer = get_mask_layer(mask=mask, mask_len=mask_len, input_dim=input_dim,
                                    init_weight=init_weight, weight_p=weight_p,
                                    scale=scale, weight_norm=weight_norm)
        if mask_layer != None:
            input_mask.append(mask_layer)
        self.input_mask = nn.Sequential(*input_mask)

        self.layer1 = Conv1dReluBn(
            input_dim, self.channels[0], kernel_size=5, padding=2)
        self.layer2 = SE_Res2Block(
            self.channels[1], kernel_size=3, stride=1, padding=2, dilation=2, scale=8)
        self.layer3 = SE_Res2Block(
            self.channels[2], kernel_size=3, stride=1, padding=3, dilation=3, scale=8)
        self.layer4 = SE_Res2Block(
            self.channels[3], kernel_size=3, stride=1, padding=4, dilation=4, scale=8)

        self.conv = nn.Conv1d(
            self.channels[4], self.channels[4], kernel_size=1)
        self.pooling = AttentiveStatsPool(self.channels[4], 128)
        self.bn0 = nn.BatchNorm1d(self.channels[4] * 2)

        self.fc1 = nn.Linear(self.channels[4] * 2, self.embedding_size)
        self.bn1 = nn.BatchNorm1d(self.embedding_size)

        self.classifier = nn.Linear(self.embedding_size, self.num_classes)

    def forward(self, x, proser=None, label=None,
                lamda_beta=0.2, mixup_alpha=-1):

        if isinstance(mixup_alpha, float) or isinstance(mixup_alpha, int):
            # layer_mix = random.randint(0, 2)
            layer_mix = mixup_alpha
        elif isinstance(mixup_alpha, list):
            layer_mix = random.choice(mixup_alpha)

        if proser != None and layer_mix == 0:
            x = self.mixup(x, proser, lamda_beta)

        x = self.input_mask(x)
        if proser != None and layer_mix == 1:
            x = self.mixup(x, proser, lamda_beta)
        if len(x.shape) == 4:
            x = x.squeeze(1).float()

        x = x.transpose(1, 2)
        out1 = self.layer1(x)
        if proser != None and layer_mix == 2:
            out1 = self.mixup(out1, proser, lamda_beta)

        out2 = self.layer2(out1) + out1
        if proser != None and layer_mix == 3:
            out2 = self.mixup(out2, proser, lamda_beta)

        out3 = self.layer3(out1 + out2) + out1 + out2
        if proser != None and layer_mix == 4:
            out3 = self.mixup(out3, proser, lamda_beta)

        out4 = self.layer4(out1 + out2 + out3) + out1 + out2 + out3
        if proser != None and layer_mix == 5:
            out4 = self.mixup(out4, proser, lamda_beta)

        out = torch.cat([out2, out3, out4], dim=1)
        out = F.relu(self.conv(out))
        out = self.bn0(self.pooling(out))
        if proser != None and layer_mix == 6:
            out = self.mixup(out, proser, lamda_beta)

        embeddings = self.bn1(self.fc1(out))

        if self.classifier == None:
            logits = ""
        else:
            logits = self.classifier(embeddings)

        # logits = self.classifier(embeddings)
        return logits, embeddings

    def xvector(self, x, embedding_type='near'):
        if len(x.shape) == 4:
            x = x.squeeze(1).float()

        if self.inst_layer != None:
            x = self.inst_layer(x)

        x = x.transpose(1, 2)
        out1 = self.layer1(x)
        out2 = self.layer2(out1) + out1
        out3 = self.layer3(out1 + out2) + out1 + out2
        out4 = self.layer4(out1 + out2 + out3) + out1 + out2 + out3

        out = torch.cat([out2, out3, out4], dim=1)
        out = F.relu(self.conv(out))
        out = self.bn0(self.pooling(out))
        if embedding_type == 'near':
            embeddings = self.fc1(out)
        else:
            embeddings = self.bn1(self.fc1(x))
        # embeddings = self.fc1(out)

        return embeddings

    def mixup(self, x, shuf_half_idx_ten, lamda_beta):
        half_batch_size = shuf_half_idx_ten.shape[0]
        half_feats = x[-half_batch_size:]
        x = torch.cat(
            [x[:-half_batch_size], lamda_beta * half_feats +
                (1 - lamda_beta) * half_feats[shuf_half_idx_ten]],
            dim=0)

        return x
# if __name__ == '__main__':
#     # Input size: batch_size * seq_len * feat_dim
#     x = torch.zeros(2, 200, 80)
#     model = ECAPA_TDNN(in_channels=80, channels=512, embd_dim=192)
#     out = model(x)
#     print(model)
#     print(out.shape)  # should be [2, 192]
