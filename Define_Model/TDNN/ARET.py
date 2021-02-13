#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: ARET.py
@Time: 2021/2/13 17:36
@Overview:
"""
import torch.nn as nn

from Define_Model.FilterLayer import L2_Norm, Mean_Norm, TimeMaskLayer, FreqMaskLayer
from Define_Model.Pooling import AttentionStatisticPooling, StatisticPooling
from Define_Model.TDNN.TDNN import TimeDelayLayer_v5


class TDNNBlock(nn.Module):

    def __init__(self, inplanes, planes, downsample=None, dilation=1, **kwargs):
        super(TDNNBlock, self).__init__()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.tdnn1 = TimeDelayLayer_v5(input_dim=inplanes, output_dim=planes, context_size=3,
                                       stride=1, dilation=dilation, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.tdnn2 = TimeDelayLayer_v5(input_dim=planes, output_dim=planes, context_size=3,
                                       stride=1, dilation=dilation, padding=1)
        # self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.tdnn1(x)
        out = self.tdnn2(out)

        # if self.downsample is not None:
        #     identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class TDNNBottleBlock(nn.Module):

    def __init__(self, inplanes, planes, downsample=None, dilation=1,
                 groups=32, **kwargs):
        super(TDNNBottleBlock, self).__init__()
        # if norm_layer is None:
        #     norm_layer = nn.BatchNorm1d
        # width = int(planes * (base_width / 32.)) * groups
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.tdnn1 = TimeDelayLayer_v5(input_dim=inplanes, output_dim=inplanes, context_size=1,
                                       stride=1, dilation=dilation)
        self.relu = nn.ReLU(inplace=True)

        self.tdnn2 = TimeDelayLayer_v5(input_dim=inplanes, output_dim=inplanes * 2, context_size=3,
                                       stride=1, dilation=dilation, padding=1, groups=groups)

        self.tdnn3 = TimeDelayLayer_v5(input_dim=inplanes * 2, output_dim=planes, context_size=1,
                                       stride=1, dilation=dilation, )

        # self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.tdnn1(x)
        out = self.tdnn2(out)
        out = self.tdnn3(out)

        # if self.downsample is not None:
        #     identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class RET(nn.Module):
    def __init__(self, num_classes, embedding_size, input_dim, alpha=0., input_norm='',
                 dropout_p=0.0, dropout_layer=False, encoder_type='STAP', block='TDNN',
                 mask='None', mask_len=20, **kwargs):
        super(RET, self).__init__()
        self.num_classes = num_classes
        self.dropout_p = dropout_p
        self.dropout_layer = dropout_layer
        self.input_dim = input_dim
        self.alpha = alpha
        self.mask = mask

        if input_norm == 'Instance':
            self.inst_layer = nn.InstanceNorm1d(input_dim)
        elif input_norm == 'Mean':
            self.inst_layer = Mean_Norm()
        else:
            self.inst_layer = None

        if self.mask == "time":
            self.maks_layer = TimeMaskLayer(mask_len=mask_len)
        elif self.mask == "freq":
            self.mask = FreqMaskLayer(mask_len=mask_len)
        elif self.mask == "time_freq":
            self.mask_layer = nn.Sequential(
                TimeMaskLayer(mask_len=mask_len),
                FreqMaskLayer(mask_len=mask_len)
            )
        else:
            self.mask_layer = None

        if block == 'Basic':
            Blocks = TDNNBlock
        elif block == 'Agg':
            Blocks = TDNNBottleBlock

        self.frame1 = TimeDelayLayer_v5(input_dim=self.input_dim, output_dim=512, context_size=5, dilation=1)
        self.frame2 = Blocks(inplanes=512, planes=512, downsample=None, dilation=1)

        self.frame4 = TimeDelayLayer_v5(input_dim=512, output_dim=512, context_size=3, dilation=1)
        self.frame5 = Blocks(inplanes=512, planes=512, downsample=None, dilation=1)

        self.frame7 = TimeDelayLayer_v5(input_dim=512, output_dim=512, context_size=3, dilation=1)
        self.frame8 = Blocks(inplanes=512, planes=512, downsample=None, dilation=1)

        self.frame10 = TimeDelayLayer_v5(input_dim=512, output_dim=512, context_size=5, dilation=1)
        self.frame11 = Blocks(inplanes=512, planes=512, downsample=None, dilation=1)

        self.frame13 = TimeDelayLayer_v5(input_dim=512, output_dim=512, context_size=1, dilation=1)
        self.frame14 = TimeDelayLayer_v5(input_dim=512, output_dim=1500, context_size=1, dilation=1)

        self.drop = nn.Dropout(p=self.dropout_p)

        if encoder_type == 'STAP':
            self.encoder = StatisticPooling(input_dim=1500)
        elif encoder_type == 'SASP':
            self.encoder = AttentionStatisticPooling(input_dim=1500, hidden_dim=512)
        else:
            raise ValueError(encoder_type)

        self.segment1 = nn.Sequential(
            nn.Linear(3000, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512)
        )

        self.segment2 = nn.Sequential(
            nn.Linear(512, embedding_size),
            nn.ReLU(),
            nn.BatchNorm1d(embedding_size)
        )

        if self.alpha:
            self.l2_norm = L2_Norm(self.alpha)

        self.classifier = nn.Linear(embedding_size, num_classes)
        # self.bn = nn.BatchNorm1d(num_classes)

        for m in self.modules():  # 对于各层参数的初始化
            if isinstance(m, nn.BatchNorm1d):  # weight设置为1，bias为0
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, TimeDelayLayer_v5):
                # nn.init.normal(m.kernel.weight, mean=0., std=1.)
                nn.init.kaiming_normal_(m.kernel.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # pdb.set_trace()
        if len(x.shape) == 4:
            x = x.squeeze(1).float()

        if self.inst_layer != None:
            x = self.inst_layer(x)

        if self.mask_layer != None:
            x = self.mask_layer(x)

        x = self.frame1(x)
        x = self.frame2(x)

        x = self.frame4(x)
        x = self.frame5(x)

        x = self.frame7(x)
        x = self.frame8(x)

        x = self.frame10(x)
        x = self.frame11(x)

        x = self.frame13(x)
        x = self.frame14(x)

        if self.dropout_layer:
            x = self.drop(x)

        # print(x.shape)
        x = self.encoder(x)
        embedding_a = self.segment1(x)
        embedding_b = self.segment2(embedding_a)

        if self.alpha:
            embedding_b = self.l2_norm(embedding_b)

        logits = self.classifier(embedding_b)

        return logits, embedding_b
