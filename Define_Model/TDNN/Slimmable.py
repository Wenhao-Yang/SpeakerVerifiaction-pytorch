#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: Slimmable.py
@Time: 2022/8/18 09:53
@Overview:
"""
import math
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from Define_Model.model import get_activation, AttrDict
from Define_Model.FilterLayer import fDLR, fBLayer, fBPLayer, fLLayer
from Define_Model.Pooling import AttentionStatisticPooling, StatisticPooling, GhostVLAD_v2, GhostVLAD_v3, \
    SelfAttentionPooling, MaxStatisticPooling
from Define_Model.FilterLayer import L2_Norm, Mean_Norm, TimeMaskLayer, FreqMaskLayer, AttentionweightLayer, \
    TimeFreqMaskLayer, AttentionweightLayer_v2, AttentionweightLayer_v0, DropweightLayer, DropweightLayer_v2, Sinc2Down, \
    Sinc2Conv, Wav2Conv, Wav2Down

FLAGS = AttrDict(width_mult_list=[1.0, 0.75, 0.50, 0.25],
                 reset_parameters=False)


class SwitchableBatchNorm2d(nn.Module):
    def __init__(self, num_features_list, width_mult_list=[1.0, 0.75, 0.50, 0.25]):
        super(SwitchableBatchNorm2d, self).__init__()
        self.num_features_list = num_features_list
        self.num_features = max(num_features_list)
        self.width_mult_list = width_mult_list

        bns = []
        for i in num_features_list:
            bns.append(nn.BatchNorm2d(i))
        self.bn = nn.ModuleList(bns)
        self.width_mult = max(width_mult_list)
        self.ignore_model_profiling = True

    def forward(self, input):
        idx = self.width_mult_list.index(self.width_mult)
        y = self.bn[idx](input)

        return y


class SlimmableConv2d(nn.Conv2d):
    def __init__(self, in_channels_list, out_channels_list,
                 kernel_size, stride=1, padding=0, dilation=1,
                 groups_list=[1], bias=True,
                 width_mult_list=[1.0, 0.75, 0.50, 0.25]):
        super(SlimmableConv2d, self).__init__(
            max(in_channels_list), max(out_channels_list),
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=max(groups_list), bias=bias)
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.groups_list = groups_list
        if self.groups_list == [1]:
            self.groups_list = [1 for _ in range(len(in_channels_list))]
        self.width_mult = max(width_mult_list)
        self.width_mult_list = width_mult_list

    def forward(self, input):
        idx = self.width_mult_list.index(self.width_mult)
        self.in_channels = self.in_channels_list[idx]
        self.out_channels = self.out_channels_list[idx]
        self.groups = self.groups_list[idx]
        weight = self.weight[:self.out_channels, :self.in_channels, :, :]
        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias
        y = nn.functional.conv2d(
            input, weight, bias, self.stride, self.padding,
            self.dilation, self.groups)
        return y


class SlimmableConv1d(nn.Conv1d):
    def __init__(self, in_channels_list, out_channels_list,
                 kernel_size, stride=1, padding=0, dilation=1,
                 groups_list=[1], bias=True,
                 width_mult_list=[1.0, 0.75, 0.50, 0.25]):
        # print(max(in_channels_list), max(out_channels_list))
        super(SlimmableConv1d, self).__init__(
            max(in_channels_list), max(out_channels_list),
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=max(groups_list), bias=bias)

        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.groups_list = groups_list
        if self.groups_list == [1]:
            self.groups_list = [1 for _ in range(len(in_channels_list))]
        self.width_mult = max(width_mult_list)
        self.width_mult_list = width_mult_list

    def forward(self, input):
        print(self.width_mult)

        idx = self.width_mult_list.index(self.width_mult)
        self.in_channels = self.in_channels_list[idx]
        self.out_channels = self.out_channels_list[idx]
        self.groups = self.groups_list[idx]
        weight = self.weight[:self.out_channels, :self.in_channels, :]
        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias
        y = nn.functional.conv1d(
            input, weight, bias, self.stride, self.padding,
            self.dilation, self.groups)
        return y


class SwitchableBatchNorm1d(nn.Module):
    def __init__(self, num_features_list, width_mult_list=[1.0, 0.75, 0.50, 0.25]):
        super(SwitchableBatchNorm1d, self).__init__()
        self.num_features_list = num_features_list
        self.num_features = max(num_features_list)
        self.width_mult_list = width_mult_list

        bns = []
        for i in num_features_list:
            bns.append(nn.BatchNorm1d(i))
        self.bn = nn.ModuleList(bns)
        self.width_mult = max(width_mult_list)
        self.ignore_model_profiling = True

    def forward(self, input):
        print(self.width_mult)

        idx = self.width_mult_list.index(self.width_mult)
        y = self.bn[idx](input)

        return y


class SlimmableLinear(nn.Linear):
    def __init__(self, in_features_list, out_features_list, bias=True,
                 width_mult_list=[1.0, 0.75, 0.50, 0.25]):
        super(SlimmableLinear, self).__init__(
            max(in_features_list), max(out_features_list), bias=bias)
        self.in_features_list = in_features_list
        self.out_features_list = out_features_list
        self.width_mult_list = width_mult_list
        self.width_mult = max(width_mult_list)

    def forward(self, input):
        idx = self.width_mult_list.index(self.width_mult)

        print(self.width_mult)
        self.in_features = self.in_features_list[idx]
        self.out_features = self.out_features_list[idx]
        weight = self.weight[:self.out_features, :self.in_features]
        if self.bias is not None:
            bias = self.bias[:self.out_features]
        else:
            bias = self.bias
        return nn.functional.linear(input, weight, bias)


def make_divisible(v, divisor=8, min_value=1):
    """
    forked from slim:
    https://github.com/tensorflow/models/blob/\
    0344c5503ee55e24f0de7f37336a6e08f10976fd/\
    research/slim/nets/mobilenet/mobilenet.py#L62-L69
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class USConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, depthwise=False, bias=True,
                 us=[True, True], ratio=[1, 1]):
        super(USConv2d, self).__init__(
            in_channels, out_channels,
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.depthwise = depthwise
        self.in_channels_max = in_channels
        self.out_channels_max = out_channels
        self.width_mult = None
        self.us = us
        self.ratio = ratio

    def forward(self, input):
        if self.us[0]:
            self.in_channels = make_divisible(
                self.in_channels_max
                * self.width_mult
                / self.ratio[0]) * self.ratio[0]
        if self.us[1]:
            self.out_channels = make_divisible(
                self.out_channels_max
                * self.width_mult
                / self.ratio[1]) * self.ratio[1]
        self.groups = self.in_channels if self.depthwise else 1
        weight = self.weight[:self.out_channels, :self.in_channels, :, :]
        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias
        y = nn.functional.conv2d(
            input, weight, bias, self.stride, self.padding,
            self.dilation, self.groups)
        if getattr(FLAGS, 'conv_averaged', False):
            y = y * (max(self.in_channels_list) / self.in_channels)

        return y


class USLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, us=[True, True]):
        super(USLinear, self).__init__(
            in_features, out_features, bias=bias)
        self.in_features_max = in_features
        self.out_features_max = out_features
        self.width_mult = None
        self.us = us

    def forward(self, input):
        if self.us[0]:
            self.in_features = make_divisible(
                self.in_features_max * self.width_mult)
        if self.us[1]:
            self.out_features = make_divisible(
                self.out_features_max * self.width_mult)
        weight = self.weight[:self.out_features, :self.in_features]
        if self.bias is not None:
            bias = self.bias[:self.out_features]
        else:
            bias = self.bias

        return nn.functional.linear(input, weight, bias)


class USBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, ratio=1):
        super(USBatchNorm2d, self).__init__(
            num_features, affine=True, track_running_stats=False)
        self.num_features_max = num_features
        # for tracking performance during training
        self.bn = nn.ModuleList([
            nn.BatchNorm2d(i, affine=False) for i in [
                make_divisible(
                    self.num_features_max * width_mult / ratio) * ratio
                for width_mult in FLAGS.width_mult_list]])
        self.ratio = ratio
        self.width_mult = None
        self.ignore_model_profiling = True

    def forward(self, input):
        weight = self.weight
        bias = self.bias
        c = make_divisible(
            self.num_features_max * self.width_mult / self.ratio) * self.ratio
        if self.width_mult in FLAGS.width_mult_list:
            idx = FLAGS.width_mult_list.index(self.width_mult)
            y = nn.functional.batch_norm(
                input,
                self.bn[idx].running_mean[:c],
                self.bn[idx].running_var[:c],
                weight[:c],
                bias[:c],
                self.training,
                self.momentum,
                self.eps)
        else:
            y = nn.functional.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                weight[:c],
                bias[:c],
                self.training,
                self.momentum,
                self.eps)
        return y


def pop_channels(autoslim_channels):
    return [i.pop(0) for i in autoslim_channels]


def bn_calibration_init(m):
    """ calculating post-statistics of batch normalization """
    if getattr(m, 'track_running_stats', False):
        # reset all values for post-statistics
        m.reset_running_stats()
        # set bn in training mode to update post-statistics
        m.training = True
        # if use cumulative moving average
        if getattr(FLAGS, 'cumulative_bn_stats', False):
            m.momentum = None


class SimmableTimeDelayLayer(nn.Module):

    def __init__(self, width_mult_list, in_channels_list=[23], out_channels_list=[512],
                 context_size=5, stride=1, dilation=1,
                 dropout_p=0.0, padding=0, groups=1, activation='relu'):
        super(SimmableTimeDelayLayer, self).__init__()
        self.context_size = context_size
        self.stride = stride
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.dilation = dilation
        self.dropout_p = dropout_p
        self.padding = padding
        self.groups = groups
        self.width_mult_list = width_mult_list

        self.kernel = [SlimmableConv1d(self.in_channels_list, self.out_channels_list,
                                       self.context_size, stride=self.stride,
                                       padding=self.padding, dilation=self.dilation,
                                       groups_list=[self.groups],
                                       width_mult_list=width_mult_list),
                       get_activation(activation)(),
                       SwitchableBatchNorm1d(out_channels_list, width_mult_list)]
        self.layers = nn.Sequential(*self.kernel)
        # self.drop = nn.Dropout(p=self.dropout_p)

    def forward(self, x):
        '''
        input: size (batch, seq_len, input_features)
        outpu: size (batch, new_seq_len, output_features)
        '''
        # _, _, d = x.shape
        #     self.input_dim, d)
        x = self.layers(x.transpose(1, 2))

        return x.transpose(1, 2)


class SlimmableTDNN(nn.Module):
    def __init__(self, num_classes, embedding_size, input_dim, alpha=0., input_norm='',
                 filter=None, sr=16000, feat_dim=64, exp=False, filter_fix=False,
                 dropout_p=0.0, dropout_layer=False, encoder_type='STAP', activation='relu',
                 num_classes_b=0, block_type='basic', first_2d=False, stride=[1],
                 init_weight='mel', power_weight=False, num_center=3, scale=0.2, weight_p=0.1,
                 mask='None', mask_len=[5, 20], channels=[512, 512, 512, 512, 1536],
                 width_mult_list=[1.0, 0.75, 0.50, 0.25], **kwargs):
        super(SlimmableTDNN, self).__init__()
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
        self.stride = stride
        self.activation = activation
        self.num_center = num_center
        self.scale = scale
        self.weight_p = weight_p

        if len(self.stride) == 1:
            while len(self.stride) < 4:
                self.stride.append(self.stride[0])
        if np.sum((self.stride)) > 4:
            print('The stride for tdnn layers are: ', str(self.stride))
        if activation != 'relu':
            print('The activation function is : ', activation)
        nonlinearity = get_activation(activation)

        if self.filter == 'fDLR':
            self.filter_layer = fDLR(input_dim=input_dim, sr=sr, num_filter=feat_dim, exp=exp, filter_fix=filter_fix)
        elif self.filter == 'fBLayer':
            self.filter_layer = fBLayer(input_dim=input_dim, sr=sr, num_filter=feat_dim, exp=exp, filter_fix=filter_fix)
        elif self.filter == 'fBPLayer':
            self.filter_layer = fBPLayer(input_dim=input_dim, sr=sr, num_filter=feat_dim, exp=exp,
                                         filter_fix=filter_fix)
        elif self.filter == 'fLLayer':
            self.filter_layer = fLLayer(input_dim=input_dim, num_filter=feat_dim, exp=exp)
        elif self.filter == 'Avg':
            self.filter_layer = nn.AvgPool2d(kernel_size=(1, 7), stride=(1, 3))
        elif self.filter == 'sinc':
            self.filter_layer = Sinc2Conv(input_dim=input_dim, out_dim=feat_dim)
        elif self.filter == 'wav2spk':
            self.filter_layer = Wav2Conv(out_dim=feat_dim)
        elif self.filter == 'sinc2down':
            self.filter_layer = Sinc2Down(input_dim, out_dim=feat_dim, fs=sr)
        elif self.filter == 'wav2down':
            self.filter_layer = Wav2Down(input_dim=input_dim, out_dim=feat_dim)
        else:
            self.filter_layer = None

        if input_norm == 'Instance':
            self.inst_layer = nn.InstanceNorm1d(input_dim)
        elif input_norm == 'Mean':
            self.inst_layer = Mean_Norm()
        else:
            self.inst_layer = None

        if self.mask == "time":
            self.maks_layer = TimeMaskLayer(mask_len=mask_len[0])
        elif self.mask == "freq":
            self.mask = FreqMaskLayer(mask_len=mask_len[0])
        elif self.mask == "both":
            self.mask_layer = TimeFreqMaskLayer(mask_len=mask_len)
        elif self.mask == "time_freq":
            self.mask_layer = nn.Sequential(
                TimeMaskLayer(mask_len=mask_len[0]),
                FreqMaskLayer(mask_len=mask_len[1])
            )
        elif self.mask == 'attention0':
            self.mask_layer = AttentionweightLayer_v0(input_dim=input_dim, weight=init_weight,
                                                      power_weight=power_weight)
        elif self.mask == 'attention':
            self.mask_layer = AttentionweightLayer(input_dim=input_dim, weight=init_weight)
        elif self.mask == 'attention2':
            self.mask_layer = AttentionweightLayer_v2(input_dim=input_dim, weight=init_weight)
        elif self.mask == 'drop':
            self.mask_layer = DropweightLayer(input_dim=input_dim, dropout_p=self.weight_p,
                                              weight=init_weight, scale=self.scale)
        elif self.mask == 'drop_v2':
            self.mask_layer = DropweightLayer_v2(input_dim=input_dim, dropout_p=self.weight_p,
                                                 weight=init_weight, scale=self.scale)
        else:
            self.mask_layer = None

        if self.filter_layer != None:
            self.input_dim = feat_dim

        if self.block_type in ['basic', 'none']:
            TDlayer = SimmableTimeDelayLayer
        else:
            raise ValueError(self.block_type)

        # in_channels_list
        self.inp = [self.input_dim for _ in FLAGS.width_mult_list]
        self.outp = [make_divisible(self.channels[0] * width_mult) for width_mult in FLAGS.width_mult_list]
        # print(self.inp, self.outp)
        self.frame1 = TDlayer(in_channels_list=self.inp, out_channels_list=self.outp,
                              context_size=5, stride=self.stride[0], activation=self.activation,
                              width_mult_list=width_mult_list)
        self.inp = self.outp
        self.outp = [make_divisible(self.channels[1] * width_mult) for width_mult in FLAGS.width_mult_list]
        # print(self.inp, self.outp)
        self.frame2 = TDlayer(in_channels_list=self.inp, out_channels_list=self.outp,
                              context_size=3, stride=self.stride[1], dilation=2, activation=self.activation,
                              width_mult_list=width_mult_list)

        self.inp = self.outp
        self.outp = [make_divisible(self.channels[2] * width_mult) for width_mult in FLAGS.width_mult_list]
        # print(self.inp, self.outp)
        self.frame3 = TDlayer(in_channels_list=self.inp, out_channels_list=self.outp,
                              context_size=3, stride=self.stride[2], dilation=3, activation=self.activation,
                              width_mult_list=width_mult_list)

        self.inp = self.outp
        self.outp = [make_divisible(self.channels[3] * width_mult) for width_mult in FLAGS.width_mult_list]
        # print(self.inp, self.outp)
        self.frame4 = TDlayer(in_channels_list=self.inp, out_channels_list=self.outp,
                              context_size=1, stride=self.stride[0], dilation=1, activation=self.activation,
                              width_mult_list=width_mult_list)

        self.inp = self.outp
        self.outp = [make_divisible(self.channels[4] * width_mult) for width_mult in FLAGS.width_mult_list]
        # print(self.inp, self.outp)
        self.frame5 = TDlayer(in_channels_list=self.inp, out_channels_list=self.outp,
                              context_size=1, stride=self.stride[3], dilation=1, activation=self.activation,
                              width_mult_list=width_mult_list)

        self.drop = nn.Dropout(p=self.dropout_p)

        if encoder_type == 'STAP':
            self.encoder = StatisticPooling(input_dim=self.channels[4])
            self.encoder_output = self.channels[4] * 2
        elif encoder_type == 'MSTAP':
            self.encoder = MaxStatisticPooling(input_dim=self.channels[4])
            self.encoder_output = self.channels[4] * 2
        elif encoder_type in ['ASTP']:
            self.encoder = AttentionStatisticPooling(input_dim=self.channels[4], hidden_dim=int(embedding_size / 2))
            self.encoder_output = self.channels[4] * 2
        elif encoder_type == 'SAP':
            self.encoder = SelfAttentionPooling(input_dim=self.channels[4], hidden_dim=self.channels[4])
            self.encoder_output = self.channels[4]
        elif encoder_type == 'Ghos_v3':
            self.encoder = GhostVLAD_v3(num_clusters=self.num_center, gost=1, dim=self.channels[4])
            self.encoder_output = self.channels[4] * 2
        else:
            raise ValueError(encoder_type)

        self.inp = [2 * x for x in self.outp]
        self.outp = [make_divisible(512 * width_mult) for width_mult in FLAGS.width_mult_list]
        # print(self.inp, self.outp)
        self.segment6 = nn.Sequential(
            SlimmableLinear(self.inp, self.outp),
            nonlinearity(),
            SwitchableBatchNorm1d(self.outp, width_mult_list)
        )

        self.segment7 = nn.Sequential(
            SlimmableLinear(self.outp, [embedding_size for _ in self.outp]),
            nonlinearity(),
            nn.BatchNorm1d(embedding_size)
        )

        if self.alpha:
            self.l2_norm = L2_Norm(self.alpha)

        if num_classes > 0:
            self.classifier = nn.Linear(embedding_size, num_classes)
        else:
            print("Set not classifier in xvectors model!")
            self.classifier = None
        # self.bn = nn.BatchNorm1d(num_classes)

        if FLAGS.reset_parameters:
            self.reset_parameters()

        # for m in self.modules():  # 对于各层参数的初始化
        #     if isinstance(m, nn.BatchNorm1d):  # weight设置为1，bias为0
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        #     elif isinstance(m, TimeDelayLayer_v5):
        #         # nn.init.normal(m.kernel.weight, mean=0., std=1.)
        #         nonlinear = 'leaky_relu' if self.activation == 'leakyrelu' else self.activation
        #         nn.init.kaiming_normal_(m.kernel.weight, mode='fan_out', nonlinearity=nonlinear)

    def set_global_dropout(self, dropout_p):
        self.dropout_p = dropout_p
        self.drop.p = dropout_p

    def forward(self, x):
        # pdb.set_trace()
        # x_vectors = self.xvector(x)
        # embedding_b = self.segment7(x_vectors)
        if self.filter_layer != None:
            x = self.filter_layer(x)

        if len(x.shape) == 4:
            x = x.squeeze(1).float()

        if self.inst_layer != None:
            x = self.inst_layer(x)

        if self.mask_layer != None:
            x = self.mask_layer(x)

        x = self.frame1(x)
        x = self.frame2(x)
        x = self.frame3(x)
        x = self.frame4(x)
        x = self.frame5(x)

        if self.dropout_layer:
            x = self.drop(x)

        # print(x.shape)
        x = self.encoder(x)
        embedding_a = self.segment6(x)
        embedding_b = self.segment7(embedding_a)

        if self.alpha:
            embedding_b = self.l2_norm(embedding_b)

        if self.classifier == None:
            logits = ""
        else:
            logits = self.classifier(embedding_b)

        return logits, embedding_b

    def xvector(self, x):
        # pdb.set_trace()
        if self.filter_layer != None:
            x = self.filter_layer(x)

        if len(x.shape) == 4:
            x = x.squeeze(1).float()

        if self.inst_layer != None:
            x = self.inst_layer(x)

        if self.mask_layer != None:
            x = self.mask_layer(x)

        # x = x.transpose(1, 2)
        x = self.frame1(x)
        x = self.frame2(x)
        x = self.frame3(x)
        x = self.frame4(x)
        x = self.frame5(x)

        if self.dropout_layer:
            x = self.drop(x)

        # print(x.shape)
        # x = self.encoder(x.transpose(1, 2))
        x = self.encoder(x)
        embedding_a = self.segment6[0](x)

        return embedding_a

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                if m.affine:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
