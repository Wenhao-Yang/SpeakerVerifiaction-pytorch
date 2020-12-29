#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: DTDNN.py
@Time: 2020/12/29 11:57
@Overview: https://github.com/yuyq96/D-TDNN/blob/master/model/layers.py
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair
import math
import torch.utils.checkpoint as cp

def get_nonlinear(config_str, channels):
    nonlinear = nn.Sequential()
    for i, name in enumerate(config_str.split('-')):
        if name == 'relu':
            nonlinear.add_module('relu', nn.ReLU(inplace=True))
        elif name == 'prelu':
            nonlinear.add_module('prelu', nn.PReLU(channels))
        elif name == 'batchnorm':
            nonlinear.add_module('batchnorm', nn.BatchNorm1d(channels))
        elif name == 'batchnorm_':
            nonlinear.add_module('batchnorm', nn.BatchNorm1d(channels, affine=False))
        else:
            raise ValueError('Unexpected module ({}).'.format(name))
    return nonlinear


class StatsPool(nn.Module):
    def __init__(self, dim=-1, keepdim=False, unbiased=True, eps=1e-2):
        super(StatsPool, self).__init__()
        self.dim = dim
        self.keepdim = keepdim
        self.unbiased = unbiased
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=self.dim)
        std = x.std(dim=self.dim, unbiased=self.unbiased)
        stats = torch.cat([mean, std], dim=-1)

        if self.keepdim:
            stats = stats.unsqueeze(dim=self.dim)
        return stats


class TransitLayer(nn.Module):

    def __init__(self, in_channels, out_channels, bias=True,
                 config_str='batchnorm-relu'):
        super(TransitLayer, self).__init__()
        self.nonlinear = get_nonlinear(config_str, in_channels)
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x):
        x = self.nonlinear(x)
        x = self.linear(x.transpose(1, 2)).transpose(1, 2)
        return x


class DenseLayer(nn.Module):

    def __init__(self, in_channels, out_channels, bias=True,
                 config_str='batchnorm-relu'):
        super(DenseLayer, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        if len(x.shape) == 2:
            x = self.linear(x)
        else:
            x = self.linear(x.transpose(1, 2)).transpose(1, 2)
        x = self.nonlinear(x)
        return x

class TimeDelay(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True):
        super(TimeDelay, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride)
        self.padding = _pair(padding)
        self.dilation = _single(dilation)
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels * kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            std = 1 / math.sqrt(self.out_channels)
            self.weight.normal_(0, std)
            if self.bias is not None:
                self.bias.normal_(0, std)

    def forward(self, x):
        x = F.pad(x, self.padding).unsqueeze(1)
        x = F.unfold(x, (self.in_channels,)+self.kernel_size, dilation=(1,)+self.dilation, stride=(1,)+self.stride)
        return F.linear(x.transpose(1, 2), self.weight, self.bias).transpose(1, 2)


class TDNNLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, bias=True, config_str='batchnorm-relu'):
        super(TDNNLayer, self).__init__()
        if padding < 0:
            assert kernel_size % 2 == 1, 'Expect equal paddings, but got even kernel size ({})'.format(kernel_size)
            padding = (kernel_size - 1) // 2 * dilation
        self.linear = TimeDelay(in_channels, out_channels, kernel_size, stride=stride,
                                padding=padding, dilation=dilation, bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        x = self.linear(x)
        x = self.nonlinear(x)
        return x


class DenseTDNNLayer(nn.Module):

    def __init__(self, in_channels, out_channels, bn_channels, kernel_size, stride=1,
                 dilation=1, bias=False, config_str='batchnorm-relu', memory_efficient=False):
        super(DenseTDNNLayer, self).__init__()
        assert kernel_size % 2 == 1, 'Expect equal paddings, but got even kernel size ({})'.format(kernel_size)
        padding = (kernel_size - 1) // 2 * dilation
        self.memory_efficient = memory_efficient
        self.nonlinear1 = get_nonlinear(config_str, in_channels)
        self.linear1 = nn.Linear(in_channels, bn_channels, bias=False)
        self.nonlinear2 = get_nonlinear(config_str, bn_channels)
        self.linear2 = TimeDelay(bn_channels, out_channels, kernel_size, stride=stride,
                                 padding=padding, dilation=dilation, bias=bias)

    def bn_function(self, x):
        return self.linear1(self.nonlinear1(x).transpose(1, 2)).transpose(1, 2)

    def forward(self, x):
        if self.training and self.memory_efficient:
            x = cp.checkpoint(self.bn_function, x)
        else:
            x = self.bn_function(x)
        x = self.linear2(self.nonlinear2(x))
        return x


class DenseTDNNBlock(nn.ModuleList):

    def __init__(self, num_layers, in_channels, out_channels, bn_channels, kernel_size,
                 stride=1, dilation=1, bias=False, config_str='batchnorm-relu', memory_efficient=False):
        super(DenseTDNNBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseTDNNLayer(
                in_channels=in_channels + i * out_channels,
                out_channels=out_channels,
                bn_channels=bn_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                bias=bias,
                config_str=config_str,
                memory_efficient=memory_efficient
            )
            self.add_module('tdnnd%d' % (i + 1), layer)

    def forward(self, x):
        for layer in self:
            x = torch.cat([x, layer(x)], 1)
        return x


class DTDNN(nn.Module):

    def __init__(self, feat_dim=30, embedding_size=512, num_classes=None,
                 growth_rate=64, bn_size=2, init_channels=128,
                 config_str='batchnorm-relu', memory_efficient=True):
        super(DTDNN, self).__init__()

        self.xvector = nn.Sequential(OrderedDict([
            ('tdnn', TDNNLayer(feat_dim, init_channels, 5, dilation=1, padding=-1,
                               config_str=config_str)),
        ]))
        channels = init_channels
        for i, (num_layers, kernel_size, dilation) in enumerate(zip((6, 12), (3, 3), (1, 3))):
            block = DenseTDNNBlock(
                num_layers=num_layers,
                in_channels=channels,
                out_channels=growth_rate,
                bn_channels=bn_size * growth_rate,
                kernel_size=kernel_size,
                dilation=dilation,
                config_str=config_str,
                memory_efficient=memory_efficient
            )
            self.xvector.add_module('block%d' % (i + 1), block)
            channels = channels + num_layers * growth_rate
            self.xvector.add_module(
                'transit%d' % (i + 1), TransitLayer(channels, channels // 2, bias=False,
                                                    config_str=config_str))
            channels //= 2
        self.xvector.add_module('stats', StatsPool())
        self.xvector.add_module('dense', DenseLayer(channels * 2, embedding_size, config_str='batchnorm_'))
        if num_classes is not None:
            self.classifier = nn.Linear(embedding_size, num_classes)

        for m in self.modules():
            if isinstance(m, TimeDelay):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.xvector(x)
        if self.training:
            x = self.classifier(x)
        return x

