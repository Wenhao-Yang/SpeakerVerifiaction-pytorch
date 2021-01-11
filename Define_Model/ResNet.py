#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: ResNet.py
@Time: 2019/10/10 下午5:09
@Overview: Deep Speaker using Resnet with CNN, which is not ordinary Resnet.
This file define resnet in 'Deep Residual Learning for Image Recognition'

For all model, the pre_forward function is for extract vectors and forward for classification.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import Bottleneck

from Define_Model.FilterLayer import TimeMaskLayer, FreqMaskLayer
from Define_Model.FilterLayer import fDLR, GRL, L2_Norm, Mean_Norm, Inst_Norm, MeanStd_Norm, CBAM
from Define_Model.Pooling import SelfAttentionPooling, AttentionStatisticPooling, StatisticPooling, AdaptiveStdPool2d, \
    SelfVadPooling, GhostVLAD_v2


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1,
                     stride=stride, bias=False)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, reduction_ratio=16):
        super(SEBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.reduction_ratio = reduction_ratio

        # Squeeze-and-Excitation
        self.glob_avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(planes, max(int(planes / self.reduction_ratio), 1))
        self.fc2 = nn.Linear(max(int(planes / self.reduction_ratio), 1), planes)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        scale = self.glob_avg(out).squeeze(dim=2).squeeze(dim=2)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.activation(scale).unsqueeze(2).unsqueeze(2)

        out = out * scale

        out += identity
        out = self.relu(out)

        return out


class CBAMBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, reduction_ratio=16):
        super(CBAMBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.reduction_ratio = reduction_ratio

        # Squeeze-and-Excitation
        self.CBAM_layer = CBAM(inplanes, planes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.CBAM_layer(out)

        out += identity
        out = self.relu(out)

        return out


class SimpleResNet(nn.Module):

    def __init__(self, block=BasicBlock,
                 num_classes=1000,
                 embedding_size=128,
                 zero_init_residual=False,
                 groups=1,
                 width_per_group=64,
                 replace_stride_with_dilation=None,
                 norm_layer=None, **kwargs):
        super(SimpleResNet, self).__init__()
        layers = [3, 4, 6, 3]
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.embedding_size=embedding_size
        self.inplanes = 16
        self.dilation = 1
        num_filter = [16, 32, 64, 128]

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(1, num_filter[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(num_filter[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        # num_filter = [16, 32, 64, 128]

        self.layer1 = self._make_layer(block, num_filter[0], layers[0])
        self.layer2 = self._make_layer(block, num_filter[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, num_filter[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, num_filter[3], layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128 * block.expansion, embedding_size)
        # self.norm = self.l2_norm(num_filter[3])
        self.alpha = 12

        self.fc2 = nn.Linear(embedding_size, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal(m.weight, mean=0., std=1.)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant(m.bn2.weight, 0)

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)

        return output

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _forward(self, x):
        # pdb.set_trace()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print(x.shape)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # pdb.set_trace()
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        x = self.l2_norm(x)
        embeddings = x * self.alpha

        x = self.fc2(embeddings)

        return x, embeddings

    # Allow for accessing forward method in a inherited class
    forward = _forward


# Analysis of Length Normalization in End-to-End Speaker Verification System
# https://arxiv.org/abs/1806.03209

class ThinResNet(nn.Module):

    def __init__(self, resnet_size=34, block=BasicBlock, inst_norm=True, kernel_size=5, stride=1, padding=2,
                 feat_dim=64, num_classes=1000, embedding_size=128, fast=False, time_dim=2, avg_size=4,
                 alpha=12, encoder_type='SAP', zero_init_residual=False, groups=1, width_per_group=64,
                 input_dim=257, sr=16000, filter=None, replace_stride_with_dilation=None, norm_layer=None,
                 input_norm='', **kwargs):
        super(ThinResNet, self).__init__()
        resnet_type = {8: [1, 1, 1, 0],
                       10: [1, 1, 1, 1],
                       18: [2, 2, 2, 2],
                       34: [3, 4, 6, 3],
                       50: [3, 4, 6, 3],
                       101: [3, 4, 23, 3]}

        layers = resnet_type[resnet_size]
        freq_dim = avg_size  # default 1
        time_dim = time_dim  # default 4
        self.inst_norm = inst_norm
        self.filter = filter
        self._norm_layer = nn.BatchNorm2d

        self.embedding_size = embedding_size
        self.inplanes = 16
        self.dilation = 1
        self.fast = fast
        num_filter = [16, 32, 64, 128]
        # num_filter = [32, 64, 128, 256]

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        if self.filter == 'fDLR':
            self.filter_layer = fDLR(input_dim=input_dim, sr=sr, num_filter=feat_dim)
        elif self.filter == 'Avg':
            self.filter_layer = nn.AvgPool2d(kernel_size=(1, 7), stride=(1, 3))
        else:
            self.filter_layer = None

        self.input_norm = input_norm
        if input_norm == 'Instance':
            self.inst_layer = nn.InstanceNorm1d(input_dim)
        elif input_norm == 'Mean':
            self.inst_layer = Mean_Norm()
        else:
            self.inst_layer = None

        self.conv1 = nn.Conv2d(1, num_filter[0], kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = self._norm_layer(num_filter[0])
        self.relu = nn.ReLU(inplace=True)

        if self.fast:
            # self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.maxpool = nn.AvgPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        self.layer1 = self._make_layer(block, num_filter[0], layers[0])
        self.layer2 = self._make_layer(block, num_filter[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, num_filter[2], layers[2], stride=2)

        if self.fast:
            self.layer4 = self._make_layer(block, num_filter[3], layers[3], stride=1)
        else:
            self.layer4 = self._make_layer(block, num_filter[3], layers[3], stride=2)

        # [64, 128, 37, 8]
        # self.avgpool = nn.AvgPool2d(kernel_size=(3, 4), stride=(2, 1))
        # 300 is the length of features

        if encoder_type == 'SAP':
            self.avgpool = nn.AdaptiveAvgPool2d((time_dim, freq_dim))
            self.encoder = SelfAttentionPooling(input_dim=num_filter[3], hidden_dim=num_filter[3])
            self.fc1 = nn.Sequential(
                nn.Linear(num_filter[3], embedding_size),
                nn.BatchNorm1d(embedding_size)
            )
        elif encoder_type == 'SASP':
            self.avgpool = nn.AdaptiveAvgPool2d((time_dim, freq_dim))
            self.encoder = AttentionStatisticPooling(input_dim=num_filter[3], hidden_dim=num_filter[3])
            self.fc1 = nn.Sequential(
                nn.Linear(num_filter[3] * 2, embedding_size),
                nn.BatchNorm1d(embedding_size)
            )
        elif encoder_type == 'STAP':
            self.avgpool = nn.AdaptiveAvgPool2d((None, freq_dim))
            self.encoder = StatisticPooling(input_dim=num_filter[3])
            self.fc1 = nn.Sequential(
                nn.Linear(num_filter[3] * 2, embedding_size),
                nn.BatchNorm1d(embedding_size)
            )
        elif encoder_type == 'ASTP':
            self.avgpool = AdaptiveStdPool2d((time_dim, freq_dim))
            self.encoder = None
            self.fc1 = nn.Sequential(
                nn.Linear(num_filter[3] * freq_dim * time_dim, embedding_size),
                nn.BatchNorm1d(embedding_size)
            )
        else:
            self.avgpool = nn.AdaptiveAvgPool2d((time_dim, freq_dim))
            self.encoder = None
            self.fc1 = nn.Sequential(
                nn.Linear(num_filter[3] * freq_dim * time_dim, embedding_size),
                nn.BatchNorm1d(embedding_size)
            )

        self.alpha = alpha
        if self.alpha:
            self.l2_norm = L2_Norm(self.alpha)

        self.classifier = nn.Linear(embedding_size, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal(m.weight, mean=0., std=1.)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)

        # Zero-initialize the last BN in each residual branch, so that the residual branch
        # starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _forward(self, x):
        # pdb.set_trace()
        # print(x.shape)
        if self.filter_layer != None:
            x = self.filter_layer(x)

        if self.inst_layer != None:
            x = self.inst_layer(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.fast:
            x = self.maxpool(x)

        # print(x.shape)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # print(x.shape)

        x = self.avgpool(x)
        if self.encoder != None:
            x = self.encoder(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        if self.alpha:
            x = self.l2_norm(x)

        logits = self.classifier(x)

        return logits, x

    # Allow for accessing forward method in a inherited class
    forward = _forward


class ResNet(nn.Module):

    def __init__(self, resnet_size=18, embedding_size=512, block=BasicBlock,
                 channels=[64, 128, 256, 512], num_classes=1000,
                 avg_size=4, zero_init_residual=False, **kwargs):
        super(ResNet, self).__init__()

        resnet_layer = {10: [1, 1, 1, 1],
                        18: [2, 2, 2, 2],
                        34: [3, 4, 6, 3],
                        50: [3, 4, 6, 3],
                        101: [3, 4, 23, 3]}

        layers = resnet_layer[resnet_size]
        self.layers = layers

        self.avg_size = avg_size
        self.channels = channels
        self.inplanes = self.channels[0]
        self.conv1 = nn.Conv2d(1, self.channels[0], kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(self.channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, self.channels[0], layers[0])
        self.layer2 = self._make_layer(block, self.channels[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, self.channels[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, self.channels[3], layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, avg_size))

        if self.layers[3] == 0:
            self.fc1 = nn.Sequential(
                nn.Linear(self.channels[2] * avg_size, embedding_size),
                nn.BatchNorm1d(embedding_size)
            )
        else:
            self.fc1 = nn.Sequential(
                nn.Linear(self.channels[3] * avg_size, embedding_size),
                nn.BatchNorm1d(embedding_size)
            )

        self.classifier = nn.Linear(embedding_size, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch, so that the residual
        # branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        if self.layers[3] != 0:
            x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        feat = self.fc1(x)
        logits = self.classifier(feat)

        return logits, feat

# model = SimpleResNet(block=BasicBlock, layers=[3, 4, 6, 3])
# input = torch.torch.randn(128,1,400,64)
# x_vectors = model.pre_forward(input)
# outputs = model(x_vectors)
# print('hello')

# M. Hajibabaei and D. Dai, “Unified hypersphere embedding for speaker recognition,”
# arXiv preprint arXiv:1807.08312, 2018.
class Block3x3(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Block3x3, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class InstBlock3x3(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(InstBlock3x3, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.InstanceNorm2d(planes)

        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.InstanceNorm2d(planes)

        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.InstanceNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class VarSizeConv(nn.Module):

    def __init__(self, inplanes, planes, stride=1, kernel_size=[3, 5, 9]):
        super(VarSizeConv, self).__init__()
        self.stide = stride

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size[0], stride=stride, padding=1)
        self.bn1 = nn.InstanceNorm2d(planes)

        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size[1], stride=stride, padding=2)
        self.bn2 = nn.InstanceNorm2d(planes)

        self.conv3 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size[2], stride=stride, padding=4)
        self.bn3 = nn.InstanceNorm2d(planes)

        self.avg = nn.AvgPool2d(kernel_size=int(stride * 2 + 1), stride=stride, padding=stride)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)

        x2 = self.conv2(x)
        x2 = self.bn2(x2)

        x3 = self.conv3(x)
        x3 = self.bn3(x3)

        if self.stide != 1:
            x = self.avg(x)

        return torch.cat([x, x1, x2, x3], dim=1)
        # return torch.cat([x, x1, x2, x3], dim=1)


class ResNet20(nn.Module):
    def __init__(self, num_classes=1000, embedding_size=128, dropout_p=0.0,
                 block=BasicBlock, input_frames=300, **kwargs):
        super(ResNet20, self).__init__()
        self.dropout_p = dropout_p
        self.inplanes = 1
        self.layer1 = self._make_layer(Block3x3, planes=64, blocks=1, stride=2)

        self.inplanes = 64
        self.layer2 = self._make_layer(Block3x3, planes=128, blocks=1, stride=2)

        self.inplanes = 128
        self.layer3 = self._make_layer(BasicBlock, 128, 1)

        self.inplanes = 128
        self.layer4 = self._make_layer(Block3x3, planes=256, blocks=1, stride=2)

        self.inplanes = 256
        self.layer5 = self._make_layer(BasicBlock, 256, 3)

        self.inplanes = 256
        self.layer6 = self._make_layer(Block3x3, planes=512, blocks=1, stride=2)

        self.inplanes = 512
        self.avgpool = nn.AdaptiveAvgPool2d((1, None))
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc1 = nn.Sequential(
            nn.Linear(17 * self.inplanes, embedding_size),
            nn.BatchNorm1d(embedding_size)
        )
        self.classifier = nn.Linear(embedding_size, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        if self.dropout_p != 0:
            x = self.dropout(x)

        feat = self.fc1(x)

        logits = self.classifier(feat)

        return logits, feat


class LocalResNet(nn.Module):
    """
    Define the ResNet model with A-softmax and AM-softmax loss.
    Added dropout as https://github.com/nagadomi/kaggle-cifar10-torch7 after average pooling and fc layer.
    """

    def __init__(self, embedding_size, num_classes, input_dim=161, block_type='basic', input_len=300,
                 relu_type='relu', resnet_size=8, channels=[64, 128, 256], dropout_p=0., encoder_type='None',
                 input_norm=None, alpha=12, stride=2, transform=False, time_dim=1, fast=False,
                 avg_size=4, kernal_size=5, padding=2, filter=None, mask='None', mask_len=25, **kwargs):

        super(LocalResNet, self).__init__()
        resnet_type = {8: [1, 1, 1, 0],
                       10: [1, 1, 1, 1],
                       14: [2, 2, 2, 0],
                       18: [2, 2, 2, 2],
                       34: [3, 4, 6, 3],
                       50: [3, 4, 6, 3],
                       101: [3, 4, 23, 3]}

        layers = resnet_type[resnet_size]

        if block_type == "seblock":
            block = SEBasicBlock
        elif block_type == 'cbam':
            block = CBAMBlock
        else:
            block = BasicBlock

        self.alpha = alpha
        self.layers = layers
        self.dropout_p = dropout_p
        self.transform = transform
        self.fast = fast
        self.mask = mask
        self.relu_type = relu_type
        self.embedding_size = embedding_size
        #
        if self.relu_type == 'relu6':
            self.relu = nn.ReLU6(inplace=True)
        elif self.relu_type == 'leakyrelu':
            self.relu = nn.LeakyReLU()
        elif self.relu_type == 'relu':
            self.relu = nn.ReLU(inplace=True)

        self.input_norm = input_norm
        self.input_len = input_len
        self.filter = filter

        if self.filter == 'Avg':
            self.filter_layer = nn.AvgPool2d(kernel_size=(1, 5), stride=(1, 2))
        else:
            self.filter_layer = None

        if input_norm == 'Instance':
            self.inst_layer = Inst_Norm(self.input_len)
        elif input_norm == 'Mean':
            self.inst_layer = Mean_Norm()
        elif input_norm == 'MeanStd':
            self.inst_layer = MeanStd_Norm()
        else:
            self.inst_layer = None

        if self.mask == "time":
            self.maks_layer = TimeMaskLayer(mask_len=mask_len)
        elif self.mask == "freq":
            self.mask = FreqMaskLayer(mask_len=mask_len)
        elif self.mask == "time_freq":
            self.mask_layer = nn.Sequential(
                TimeMaskLayer(),
                FreqMaskLayer()
            )
        else:
            self.mask_layer = None

        self.inplanes = channels[0]
        self.conv1 = nn.Conv2d(1, channels[0], kernel_size=(5, 5), stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(channels[0])
        if self.fast:
            # self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.maxpool = nn.AvgPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        # self.maxpool = nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.layer1 = self._make_layer(block, channels[0], layers[0])

        self.inplanes = channels[1]
        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=kernal_size, stride=2,
                               padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(channels[1])
        self.layer2 = self._make_layer(block, channels[1], layers[1])

        self.inplanes = channels[2]
        self.conv3 = nn.Conv2d(channels[1], channels[2], kernel_size=kernal_size, stride=2,
                               padding=padding, bias=False)
        self.bn3 = nn.BatchNorm2d(channels[2])
        self.layer3 = self._make_layer(block, channels[2], layers[2])

        if layers[3] != 0:
            assert len(channels) == 4
            self.inplanes = channels[3]
            if self.fast:
                self.conv4 = nn.Conv2d(channels[2], channels[3], kernel_size=kernal_size, stride=1,
                                       padding=padding, bias=False)
            else:
                self.conv4 = nn.Conv2d(channels[2], channels[3], kernel_size=kernal_size, stride=2,
                                       padding=padding, bias=False)
            self.bn4 = nn.BatchNorm2d(channels[3])
            self.layer4 = self._make_layer(block=block, planes=channels[3], blocks=layers[3])

        self.dropout = nn.Dropout(self.dropout_p)

        if encoder_type == 'SAP':
            self.avgpool = nn.AdaptiveAvgPool2d((time_dim, avg_size))
            self.encoder = SelfAttentionPooling(input_dim=channels[-1], hidden_dim=channels[-1])
            self.fc1 = nn.Sequential(
                nn.Linear(channels[-1], embedding_size),
                nn.ReLU(),
                nn.BatchNorm1d(embedding_size)
            )
        elif encoder_type == 'SASP':
            self.avgpool = nn.AdaptiveAvgPool2d((time_dim, avg_size))
            self.encoder = AttentionStatisticPooling(input_dim=channels[-1], hidden_dim=channels[-1])
            self.fc1 = nn.Sequential(
                nn.Linear(channels[-1] * 2, embedding_size),
                nn.ReLU(),
                nn.BatchNorm1d(embedding_size)
            )
        elif encoder_type == 'STAP':
            self.avgpool = nn.AdaptiveAvgPool2d((None, avg_size))
            self.encoder = StatisticPooling(input_dim=avg_size*channels[-1])
            self.fc1 = nn.Sequential(
                nn.Linear(avg_size * channels[-1] * 2, embedding_size),
                nn.ReLU(),
                nn.BatchNorm1d(embedding_size)
            )
        elif encoder_type == 'ASTP':
            self.avgpool = AdaptiveStdPool2d((time_dim, avg_size))
            self.encoder = None
            self.fc1 = nn.Sequential(
                nn.Linear(channels[-1] * avg_size * time_dim, embedding_size),
                nn.ReLU(),
                nn.BatchNorm1d(embedding_size)
            )
        else:
            self.avgpool = nn.AdaptiveAvgPool2d((time_dim, avg_size))
            self.encoder = None
            self.fc1 = nn.Sequential(
                nn.Linear(channels[-1] * avg_size * time_dim, embedding_size),
                nn.ReLU(),
                nn.BatchNorm1d(embedding_size)
            )

        if self.transform == 'Linear':
            self.trans_layer = nn.Sequential(
                nn.Linear(embedding_size, embedding_size),
                nn.ReLU(),
                nn.BatchNorm1d(embedding_size)
            )
        elif self.transform == 'GhostVLAD':
            self.trans_layer = GhostVLAD_v2(num_clusters=8, gost=1, dim=embedding_size, normalize_input=True)
        else:
            self.trans_layer = None

        if self.alpha:
            self.l2_norm = L2_Norm(self.alpha)

        # self.fc = nn.Linear(self.inplanes * avg_size, embedding_size)
        self.classifier = nn.Linear(self.embedding_size, num_classes)

        for m in self.modules():  # 对于各层参数的初始化
            if isinstance(m, nn.Conv2d):  # 以2/n的开方为标准差，做均值为0的正态分布
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):  # weight设置为1，bias为0
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.filter_layer != None:
            x = self.filter_layer(x)

        if self.inst_layer != None:
            x = self.inst_layer(x)

        if self.mask_layer != None:
            x = self.mask_layer(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.fast:
            x = self.maxpool(x)
        x = self.layer1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.layer3(x)

        if self.layers[3] != 0:
            x = self.conv4(x)
            x = self.bn4(x)
            x = self.relu(x)
            x = self.layer4(x)

        if self.dropout_p > 0:
            x = self.dropout(x)

        x = self.avgpool(x)
        if self.encoder != None:
            x = self.encoder(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        if self.trans_layer != None:
            x = self.trans_layer(x)
            # x = t_x + x

        if self.alpha:
            x = self.l2_norm(x)

        logits = self.classifier(x)

        return logits, x


# previoud version for test
# class LocalResNet(nn.Module):
#     """
#     Define the ResNet model with A-softmax and AM-softmax loss.
#     Added dropout as https://github.com/nagadomi/kaggle-cifar10-torch7 after average pooling and fc layer.
#     """
#
#     def __init__(self, embedding_size, num_classes,
#                  input_dim=161, block=BasicBlock,
#                  resnet_size=8, channels=[64, 128, 256], dropout_p=0.,
#                  inst_norm=False, alpha=12, stride=2, transform=False,
#                  avg_size=4, kernal_size=5, padding=2, **kwargs):
#
#         super(LocalResNet, self).__init__()
#         resnet_type = {8: [1, 1, 1, 0],
#                        10: [1, 1, 1, 1],
#                        18: [2, 2, 2, 2],
#                        34: [3, 4, 6, 3],
#                        50: [3, 4, 6, 3],
#                        101: [3, 4, 23, 3]}
#
#         layers = resnet_type[resnet_size]
#         self.alpha = alpha
#         self.layers = layers
#         self.dropout_p = dropout_p
#         self.transform = transform
#
#         self.embedding_size = embedding_size
#         # self.relu = nn.LeakyReLU()
#         self.relu = nn.ReLU(inplace=True)
#         self.inst_norm = inst_norm
#         self.inst_layer = nn.InstanceNorm1d(input_dim)
#
#         self.inplanes = channels[0]
#         self.conv1 = nn.Conv2d(1, channels[0], kernel_size=(5, 5), stride=stride, padding=(3, 2))
#         self.bn1 = nn.BatchNorm2d(channels[0])
#         self.maxpool = nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
#
#         self.layer1 = self._make_layer(block, channels[0], layers[0])
#
#         self.inplanes = channels[1]
#         self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=kernal_size, stride=2,
#                                padding=padding, bias=False)
#         self.bn2 = nn.BatchNorm2d(channels[1])
#         self.layer2 = self._make_layer(block, channels[1], layers[1])
#
#         self.inplanes = channels[2]
#         self.conv3 = nn.Conv2d(channels[1], channels[2], kernel_size=kernal_size, stride=2,
#                                padding=padding, bias=False)
#         self.bn3 = nn.BatchNorm2d(channels[2])
#         self.layer3 = self._make_layer(block, channels[2], layers[2])
#
#         if layers[3] != 0:
#             assert len(channels) == 4
#             self.inplanes = channels[3]
#             self.conv4 = nn.Conv2d(channels[2], channels[3], kernel_size=kernal_size, stride=2,
#                                    padding=padding, bias=False)
#             self.bn4 = nn.BatchNorm2d(channels[3])
#             self.layer4 = self._make_layer(block=block, planes=channels[3], blocks=layers[3])
#
#         self.dropout = nn.Dropout(self.dropout_p)
#         self.avg_pool = nn.AdaptiveAvgPool2d((1, avg_size))
#
#         self.fc = nn.Sequential(
#             nn.Linear(self.inplanes * avg_size, embedding_size),
#             nn.BatchNorm1d(embedding_size)
#         )
#
#         if self.transform:
#             self.trans_layer = nn.Sequential(
#                 nn.Linear(embedding_size, embedding_size, bias=False),
#                 nn.BatchNorm1d(embedding_size),
#                 nn.ReLU()
#             )
#
#         # self.fc = nn.Linear(self.inplanes * avg_size, embedding_size)
#         self.classifier = nn.Linear(self.embedding_size, num_classes)
#
#         for m in self.modules():  # 对于各层参数的初始化
#             if isinstance(m, nn.Conv2d):  # 以2/n的开方为标准差，做均值为0的正态分布
#                 # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 # m.weight.data.normal_(0, math.sqrt(2. / n))
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):  # weight设置为1，bias为0
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#     def l2_norm(self, input, alpha=1.0):
#         # alpha = log(p * ( class -2) / (1-p))
#         input_size = input.size()
#         buffer = torch.pow(input, 2)
#
#         normp = torch.sum(buffer, 1).add_(1e-12)
#         norm = torch.sqrt(normp)
#
#         _output = torch.div(input, norm.view(-1, 1).expand_as(input))
#         output = _output.view(input_size)
#         # # # input = input.renorm(p=2, dim=1, maxnorm=1.0)
#         # norm = input.norm(p=2, dim=1, keepdim=True).add(1e-14)
#         # output = input / norm
#
#         return output * alpha
#
#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 conv1x1(self.inplanes, planes * block.expansion, stride),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.inplanes, planes))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         if self.inst_norm:
#             x = x.squeeze(1).transpose(1, 2)
#             x = self.inst_layer(x)
#             x = x.transpose(1, 2).unsqueeze(1)
#
#             # x = x - torch.mean(x, dim=-2, keepdim=True)
#
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#         x = self.layer2(x)
#
#         x = self.conv3(x)
#         x = self.bn3(x)
#         x = self.relu(x)
#         x = self.layer3(x)
#
#         if self.layers[3] != 0:
#             x = self.conv4(x)
#             x = self.bn4(x)
#             x = self.relu(x)
#             x = self.layer4(x)
#
#         if self.dropout_p > 0:
#             x = self.dropout(x)
#
#         # if self.statis_pooling:
#         #     mean_x = self.avg_pool(x)
#         #     mean_x = mean_x.view(mean_x.size(0), -1)
#         #
#         #     std_x = self.std_pool(x)
#         #     std_x = std_x.view(std_x.size(0), -1)
#         #
#         #     x = torch.cat((mean_x, std_x), dim=1)
#         #
#         # else:
#         # print(x.shape)
#         x = self.avg_pool(x)
#         x = x.view(x.size(0), -1)
#
#         x = self.fc(x)
#         if self.transform == True:
#             x += self.trans_layer(x)
#             t_x = self.trans_layer(x)
#             x = t_x + x
#
#         if self.alpha:
#             x = self.l2_norm(x, alpha=self.alpha)
#
#         logits = self.classifier(x)
#
#         return logits, x


class DomainResNet(nn.Module):
    """
    Define the ResNet model with A-softmax and AM-softmax loss.
    Added dropout as https://github.com/nagadomi/kaggle-cifar10-torch7 after average pooling and fc layer.
    """

    def __init__(self, embedding_size_a, embedding_size_b, embedding_size_o,
                 num_classes_a, num_classes_b,
                 block=BasicBlock, input_dim=161,
                 resnet_size=8, channels=[64, 128, 256], dropout_p=0.,
                 inst_norm=False, alpha=12,
                 avg_size=4, kernal_size=5, padding=2, **kwargs):

        super(DomainResNet, self).__init__()
        resnet_type = {8: [1, 1, 1, 0],
                       10: [1, 1, 1, 1],
                       18: [2, 2, 2, 2],
                       34: [3, 4, 6, 3],
                       50: [3, 4, 6, 3],
                       101: [3, 4, 23, 3]}

        layers = resnet_type[resnet_size]
        self.alpha = alpha
        self.layers = layers
        self.dropout_p = dropout_p

        self.embedding_size_a = embedding_size_a
        self.embedding_size_b = embedding_size_b
        self.embedding_size = embedding_size_a + embedding_size_b - embedding_size_o

        # self.relu = nn.LeakyReLU()
        self.relu = nn.ReLU(inplace=True)

        self.inst_norm = inst_norm
        # self.inst_layer = nn.InstanceNorm1d(input_dim)

        self.inplanes = channels[0]
        self.conv1 = nn.Conv2d(1, channels[0], kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])

        self.maxpool = nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=1)
        self.layer1 = self._make_layer(block, channels[0], layers[0])

        self.inplanes = channels[1]
        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=kernal_size, stride=2,
                               padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(channels[1])
        self.layer2 = self._make_layer(block, channels[1], layers[1])

        self.inplanes = channels[2]
        self.conv3 = nn.Conv2d(channels[1], channels[2], kernel_size=kernal_size, stride=2,
                               padding=padding, bias=False)
        self.bn3 = nn.BatchNorm2d(channels[2])
        self.layer3 = self._make_layer(block, channels[2], layers[2])

        if layers[3] != 0:
            assert len(channels) == 4
            self.inplanes = channels[3]
            self.conv4 = nn.Conv2d(channels[2], channels[3], kernel_size=kernal_size, stride=2,
                                   padding=padding, bias=False)
            self.bn4 = nn.BatchNorm2d(channels[3])
            self.layer4 = self._make_layer(block=block, planes=channels[3], blocks=layers[3])

        self.dropout = nn.Dropout(self.dropout_p)
        self.avg_pool = nn.AdaptiveAvgPool2d((avg_size, 1))

        # self.encoder = nn.LSTM(input_size=channels[2],
        #                        hidden_size=channels[2],
        #                        num_layers=1,
        #                        batch_first=True,
        #                        dropout=self.dropout_p)

        self.fc = nn.Sequential(
            nn.Linear(self.inplanes * avg_size, self.embedding_size),
            nn.BatchNorm1d(self.embedding_size)
        )

        self.classifier_spk = nn.Linear(self.embedding_size_a, num_classes_a)

        self.grl = GRL(lambda_=0.)
        self.classifier_dom = nn.Sequential(nn.Linear(self.embedding_size_b, int(self.embedding_size_b / 4)),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(int(self.embedding_size_b / 4), num_classes_b),
                                            )


        for m in self.modules():  # 对于各层参数的初始化
            if isinstance(m, nn.Conv2d):  # 以2/n的开方为标准差，做均值为0的正态分布
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):  # weight设置为1，bias为0
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def l2_norm(self, input, alpha=1.0):
        # alpha = log(p * (
        #
        # class -2) / (1-p))
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-12)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        # # # input = input.renorm(p=2, dim=1, maxnorm=1.0)
        #
        # norm = input.norm(p=2, dim=1, keepdim=True).add(1e-14)
        # output = input / norm

        return output * alpha

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.inst_norm:
            x = x.squeeze(1)
            x = self.inst_layer(x)
            x = x.unsqueeze(1)
            # x = x - torch.mean(x, dim=-2, keepdim=True)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.layer3(x)

        if self.layers[3] != 0:
            x = self.conv4(x)
            x = self.bn4(x)
            x = self.relu(x)
            x = self.layer4(x)

        if self.dropout_p > 0:
            x = self.dropout(x)
        # x = self.avg_pool(x).transpose(1, 2)
        # x, (_, _) = self.encoder(x.squeeze(1))
        # x = x[:, -1]
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        spk_x = x[:, :self.embedding_size_a]
        dom_x = x[:, -self.embedding_size_b:]

        if self.alpha:
            spk_x = self.l2_norm(spk_x, alpha=self.alpha)

        spk_logits = self.classifier_spk(spk_x)

        dom_x = self.grl(dom_x)
        dom_logits = self.classifier_dom(dom_x)

        return spk_logits, spk_x, dom_logits, dom_x


class GradResNet(nn.Module):
    """
    Define the ResNet model with A-softmax and AM-softmax loss.
    Added dropout as https://github.com/nagadomi/kaggle-cifar10-torch7 after average pooling and fc layer.
    """

    def __init__(self, embedding_size, num_classes, block=BasicBlock, input_dim=161,
                 resnet_size=8, channels=[64, 128, 256], dropout_p=0., ince=False, transform=False,
                 inst_norm=False, alpha=12, vad=False, avg_size=4, kernal_size=5, padding=2, **kwargs):

        super(GradResNet, self).__init__()
        resnet_type = {8: [1, 1, 1, 0],
                       10: [1, 1, 1, 1],
                       18: [2, 2, 2, 2],
                       34: [3, 4, 6, 3],
                       50: [3, 4, 6, 3],
                       101: [3, 4, 23, 3]}

        layers = resnet_type[resnet_size]
        self.ince = ince
        self.alpha = alpha
        self.layers = layers
        self.dropout_p = dropout_p
        self.transform = transform

        self.embedding_size = embedding_size
        # self.relu = nn.LeakyReLU()
        self.relu = nn.ReLU(inplace=True)
        self.vad = vad
        if self.vad:
            self.vad_layer = SelfVadPooling(input_dim)

        self.inst_norm = inst_norm
        # self.inst_layer = nn.InstanceNorm1d(input_dim)

        if self.ince:
            self.pre_conv = VarSizeConv(1, 1)
            self.conv1 = nn.Conv2d(4, channels[0], kernel_size=5, stride=2, padding=2)
        else:
            self.conv1 = nn.Conv2d(1, channels[0], kernel_size=5, stride=2, padding=2)

        self.bn1 = nn.BatchNorm2d(channels[0])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inplanes = channels[0]
        self.layer1 = self._make_layer(block, channels[0], layers[0])

        self.inplanes = channels[1]
        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=kernal_size,
                               stride=2, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(channels[1])
        self.layer2 = self._make_layer(block, channels[1], layers[1])

        self.inplanes = channels[2]
        self.conv3 = nn.Conv2d(channels[1], channels[2], kernel_size=kernal_size,
                               stride=2, padding=padding, bias=False)
        self.bn3 = nn.BatchNorm2d(channels[2])
        self.layer3 = self._make_layer(block, channels[2], layers[2])

        if layers[3] != 0:
            assert len(channels) == 4
            self.inplanes = channels[3]
            self.conv4 = nn.Conv2d(channels[2], channels[3], kernel_size=kernal_size, stride=2,
                                   padding=padding, bias=False)
            self.bn4 = nn.BatchNorm2d(channels[3])
            self.layer4 = self._make_layer(block=block, planes=channels[3], blocks=layers[3])

        self.dropout = nn.Dropout(self.dropout_p)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, avg_size))

        if self.transform:
            self.trans_layer = nn.Sequential(
                nn.Linear(embedding_size, embedding_size, bias=False),
                nn.BatchNorm1d(embedding_size),
                nn.ReLU()
            )

        self.fc = nn.Sequential(
            nn.Linear(self.inplanes * avg_size, embedding_size),
            nn.BatchNorm1d(embedding_size)
        )

        # self.fc = nn.Linear(self.inplanes * avg_size, embedding_size)
        self.classifier = nn.Linear(self.embedding_size, num_classes)

        for m in self.modules():  # 对于各层参数的初始化
            if isinstance(m, nn.Conv2d):  # 以2/n的开方为标准差，做均值为0的正态分布
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):  # weight设置为1，bias为0
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def l2_norm(self, input, alpha=1.0):
        # alpha = log(p * (class -2) / (1-p))
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-12)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        # # # input = input.renorm(p=2, dim=1, maxnorm=1.0)
        # norm = input.norm(p=2, dim=1, keepdim=True).add(1e-14)
        # output = input / norm

        return output * alpha

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.vad:
            x = self.vad_layer(x)

        x = torch.log(x)

        if self.inst_norm:
            # x = self.inst_layer(x)
            x = x - torch.mean(x, dim=-2, keepdim=True)

        if self.ince:
            x = self.pre_conv(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.layer3(x)

        if self.layers[3] != 0:
            x = self.conv4(x)
            x = self.bn4(x)
            x = self.relu(x)
            x = self.layer4(x)

        if self.dropout_p > 0:
            x = self.dropout(x)

        # if self.statis_pooling:
        #     mean_x = self.avg_pool(x)
        #     mean_x = mean_x.view(mean_x.size(0), -1)
        #
        #     std_x = self.std_pool(x)
        #     std_x = std_x.view(std_x.size(0), -1)
        #
        #     x = torch.cat((mean_x, std_x), dim=1)
        #
        # else:
        # print(x.shape)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        if self.transform:
            t_x = self.trans_layer(x)
            x = t_x + x

        if self.alpha:
            x = self.l2_norm(x, alpha=self.alpha)

        logits = self.classifier(x)

        return logits, x


class TimeFreqResNet(nn.Module):
    """
    Define the ResNet model with A-softmax and AM-softmax loss.
    Added dropout as https://github.com/nagadomi/kaggle-cifar10-torch7 after average pooling and fc layer.
    """

    def __init__(self, embedding_size, num_classes, block=BasicBlock, input_dim=161,
                 resnet_size=8, channels=[64, 128, 256], dropout_p=0., ince=False,
                 inst_norm=False, alpha=12, vad=False, avg_size=4, kernal_size=5, padding=2, **kwargs):

        super(TimeFreqResNet, self).__init__()
        resnet_type = {8: [1, 1, 1, 0],
                       10: [1, 1, 1, 1],
                       18: [2, 2, 2, 2],
                       34: [3, 4, 6, 3],
                       50: [3, 4, 6, 3],
                       101: [3, 4, 23, 3]}

        layers = resnet_type[resnet_size]
        self.ince = ince
        self.alpha = alpha
        self.layers = layers
        self.dropout_p = dropout_p

        self.embedding_size = embedding_size
        # self.relu = nn.LeakyReLU()
        self.relu = nn.ReLU(inplace=True)
        self.vad = vad
        if self.vad:
            self.vad_layer = SelfVadPooling(input_dim)

        self.inst_norm = inst_norm
        # self.inst_layer = nn.InstanceNorm1d(input_dim)

        self.conv1 = nn.Sequential(nn.Conv2d(1, channels[0], kernel_size=(5, 1), stride=(2, 1), padding=(2, 0)),
                                   nn.BatchNorm2d(channels[0]),
                                   nn.Conv2d(channels[0], channels[0], kernel_size=(1, 5), stride=(1, 2),
                                             padding=(0, 2)),
                                   )


        self.bn1 = nn.BatchNorm2d(channels[0])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inplanes = channels[0]
        self.layer1 = self._make_layer(block, channels[0], layers[0])

        self.inplanes = channels[1]

        # self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=kernal_size,
        #                        stride=2, padding=padding, bias=False)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], kernel_size=(5, 1), stride=(2, 1), padding=(2, 0)),
            nn.BatchNorm2d(channels[1]),
            nn.Conv2d(channels[1], channels[1], kernel_size=(1, 5), stride=(1, 2),
                      padding=(0, 2)),
            )

        self.bn2 = nn.BatchNorm2d(channels[1])
        self.layer2 = self._make_layer(block, channels[1], layers[1])

        self.inplanes = channels[2]
        # self.conv3 = nn.Conv2d(channels[1], channels[2], kernel_size=kernal_size,
        #                        stride=2, padding=padding, bias=False)
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], kernel_size=(5, 1), stride=(2, 1), padding=(2, 0)),
            nn.BatchNorm2d(channels[2]),
            nn.Conv2d(channels[2], channels[2], kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
        )

        self.bn3 = nn.BatchNorm2d(channels[2])
        self.layer3 = self._make_layer(block, channels[2], layers[2])

        if layers[3] != 0:
            assert len(channels) == 4
            self.inplanes = channels[3]
            self.conv4 = nn.Conv2d(channels[2], channels[3], kernel_size=kernal_size, stride=2,
                                   padding=padding, bias=False)
            self.bn4 = nn.BatchNorm2d(channels[3])
            self.layer4 = self._make_layer(block=block, planes=channels[3], blocks=layers[3])

        self.dropout = nn.Dropout(self.dropout_p)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, avg_size))

        self.fc = nn.Sequential(
            nn.Linear(self.inplanes * avg_size, embedding_size),
            nn.BatchNorm1d(embedding_size)
        )
        # self.fc = nn.Linear(self.inplanes * avg_size, embedding_size)
        self.classifier = nn.Linear(self.embedding_size, num_classes)

        for m in self.modules():  # 对于各层参数的初始化
            if isinstance(m, nn.Conv2d):  # 以2/n的开方为标准差，做均值为0的正态分布
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):  # weight设置为1，bias为0
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def l2_norm(self, input, alpha=1.0):
        # alpha = log(p * (class -2) / (1-p))
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-12)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)

        return output * alpha

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.vad:
            x = self.vad_layer(x)

        x = torch.log(x)

        if self.inst_norm:
            # x = self.inst_layer(x)
            x = x - torch.mean(x, dim=-2, keepdim=True)

        if self.ince:
            x = self.pre_conv(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.layer3(x)

        if self.layers[3] != 0:
            x = self.conv4(x)
            x = self.bn4(x)
            x = self.relu(x)
            x = self.layer4(x)

        if self.dropout_p > 0:
            x = self.dropout(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        if self.alpha:
            x = F.self.l2_norm(x, alpha=self.alpha)

        logits = self.classifier(x)

        return logits, x


class MultiResNet(nn.Module):
    """
    Define the ResNet model with A-softmax and AM-softmax loss.
    Added dropout as https://github.com/nagadomi/kaggle-cifar10-torch7 after average pooling and fc layer.
    """

    def __init__(self, embedding_size, num_classes_a, num_classes_b, block=BasicBlock, input_dim=161,
                 resnet_size=8, channels=[64, 128, 256], dropout_p=0., stride=2, fast=False,
                 inst_norm=False, alpha=12, input_norm='None', transform=False,
                 avg_size=4, kernal_size=5, padding=2, mask='None', mask_len=25, **kwargs):

        super(MultiResNet, self).__init__()
        resnet_type = {8: [1, 1, 1, 0],
                       10: [1, 1, 1, 1],
                       18: [2, 2, 2, 2],
                       34: [3, 4, 6, 3],
                       50: [3, 4, 6, 3],
                       101: [3, 4, 23, 3]}

        layers = resnet_type[resnet_size]
        self.alpha = alpha
        self.layers = layers
        self.dropout_p = dropout_p
        self.embedding_size = embedding_size
        self.relu = nn.ReLU(inplace=True)
        self.transform = transform
        self.fast = fast
        self.input_norm = input_norm
        self.mask = mask

        if input_norm == 'Instance':
            self.inst_layer = nn.InstanceNorm1d(input_dim)
        elif input_norm == 'Mean':
            self.inst_layer = Mean_Norm()
        elif input_norm == 'MeanStd':
            self.inst_layer = MeanStd_Norm()
        else:
            self.inst_layer = None

        if self.mask == "time":
            self.maks_layer = TimeMaskLayer(mask_len=mask_len)
        elif self.mask == "freq":
            self.mask_layer = FreqMaskLayer(mask_len=mask_len)
        elif self.mask == "time_freq":
            self.mask_layer = nn.Sequential(
                TimeMaskLayer(mask_len=mask_len),
                FreqMaskLayer(mask_len=mask_len)
            )
        else:
            self.mask_layer = None

        self.inplanes = channels[0]
        self.conv1 = nn.Conv2d(1, channels[0], kernel_size=5, stride=stride, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])

        if self.fast:
            self.maxpool = nn.AvgPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        else:
            self.maxpool = None

        # self.maxpool = nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=1)
        self.layer1 = self._make_layer(block, channels[0], layers[0])

        self.inplanes = channels[1]
        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=kernal_size, stride=2,
                               padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(channels[1])
        self.layer2 = self._make_layer(block, channels[1], layers[1])

        self.inplanes = channels[2]
        self.conv3 = nn.Conv2d(channels[1], channels[2], kernel_size=kernal_size, stride=2,
                               padding=padding, bias=False)
        self.bn3 = nn.BatchNorm2d(channels[2])
        self.layer3 = self._make_layer(block, channels[2], layers[2])

        if layers[3] != 0:
            assert len(channels) == 4
            self.inplanes = channels[3]
            self.conv4 = nn.Conv2d(channels[2], channels[3], kernel_size=kernal_size, stride=2,
                                   padding=padding, bias=False)
            self.bn4 = nn.BatchNorm2d(channels[3])
            self.layer4 = self._make_layer(block=block, planes=channels[3], blocks=layers[3])

        self.dropout = nn.Dropout(self.dropout_p)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, avg_size))

        self.fc = nn.Sequential(
            nn.Linear(self.inplanes * avg_size, self.embedding_size),
            nn.BatchNorm1d(self.embedding_size)
        )
        if self.transform == 'Linear':
            self.trans_layer = nn.Sequential(
                nn.Linear(embedding_size, embedding_size),
                nn.ReLU(),
                nn.BatchNorm1d(embedding_size))
        elif self.transform == 'GhostVLAD':
            self.trans_layer = GhostVLAD_v2(num_clusters=8, gost=1, dim=embedding_size, normalize_input=True)
        else:
            self.trans_layer = None
        if self.alpha:
            self.l2_norm = L2_Norm(self.alpha)

        self.classifier_a = nn.Linear(self.embedding_size, num_classes_a)
        self.classifier_b = nn.Linear(self.embedding_size, num_classes_b)

        for m in self.modules():  # 对于各层参数的初始化
            if isinstance(m, nn.Conv2d):  # 以2/n的开方为标准差，做均值为0的正态分布
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):  # weight设置为1，bias为0
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        tuple_input = False
        if isinstance(x, tuple):
            tuple_input = True
            size_a = len(x[0])
            x = torch.cat(x, dim=0)

        if self.inst_layer != None:
            x = self.inst_layer(x)

        if self.mask_layer != None:
            x = self.mask_layer(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.maxpool != None:
            x = self.maxpool(x)

        x = self.layer1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.layer3(x)

        if self.layers[3] != 0:
            x = self.conv4(x)
            x = self.bn4(x)
            x = self.relu(x)
            x = self.layer4(x)

        if self.dropout_p > 0:
            x = self.dropout(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)

        embeddings = self.fc(x)

        if self.trans_layer != None:
            embeddings = self.trans_layer(embeddings)

        if self.alpha:
            embeddings = self.l2_norm(embeddings)
            # embeddings = self.l2_norm(embeddings, alpha=self.alpha)

        if tuple_input:
            embeddings_a = embeddings[:size_a]
            embeddings_b = embeddings[size_a:]

            logits_a = self.classifier_a(embeddings_a)
            logits_b = self.classifier_b(embeddings_b)

            return (logits_a, logits_b), (embeddings_a, embeddings_b)
        else:
            return '', embeddings

    # def cls_forward(self, a, b):
    #
    #     logits_a = self.classifier_a(a)
    #     logits_b = self.classifier_b(b)
    #
    #     return logits_a, logits_b
