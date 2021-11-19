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
import torch
import torch.nn as nn

from Define_Model.FilterLayer import L2_Norm, Mean_Norm, TimeMaskLayer, FreqMaskLayer
from Define_Model.Pooling import AttentionStatisticPooling, StatisticPooling
from Define_Model.TDNN.TDNN import TimeDelayLayer_v5, TimeDelayLayer_v6, ShuffleTDLayer, channel_shuffle
from Define_Model.model import get_activation


class TDNNBlock(nn.Module):

    def __init__(self, inplanes, planes, downsample=None, dilation=1, **kwargs):
        super(TDNNBlock, self).__init__()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        if isinstance(downsample, int):
            inter_connect = int(planes / downsample)
        else:
            inter_connect = planes

        self.tdnn1 = TimeDelayLayer_v5(input_dim=inplanes, output_dim=inter_connect, context_size=3,
                                       stride=1, dilation=dilation, padding=1)
        # self.relu = nn.ReLU(inplace=True)
        self.tdnn2 = TimeDelayLayer_v5(input_dim=inter_connect, output_dim=planes, context_size=3,
                                       stride=1, dilation=dilation, padding=1)
        # self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.tdnn1(x)
        out = self.tdnn2(out)

        out += identity
        # out = self.relu(out)

        return out


class TDNNBlock_v2(nn.Module):

    def __init__(self, inplanes, planes, downsample=None, dilation=1, activation='relu', **kwargs):
        super(TDNNBlock_v2, self).__init__()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        if isinstance(downsample, int):
            inter_connect = int(planes / downsample)
        else:
            inter_connect = planes

        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation in ['leakyrelu', 'leaky_relu']:
            act_fn = nn.LeakyReLU
        elif activation == 'prelu':
            act_fn = nn.PReLU

        self.tdnn1_kernel = nn.Conv1d(inplanes, inter_connect, 3, stride=1,
                                      padding=1, dilation=dilation, bias=False)
        self.tdnn1_bn = nn.BatchNorm1d(inter_connect)
        self.act = act_fn()

        self.tdnn2_kernel = nn.Conv1d(inter_connect, planes, 3, stride=1,
                                      padding=1, dilation=dilation, bias=False)
        self.tdnn2_bn = nn.BatchNorm1d(planes)

        # self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.tdnn1_kernel(x.transpose(1, 2))
        out = self.tdnn1_bn(out)
        out = self.act(out)

        out = self.tdnn2_kernel(out)
        out = self.tdnn2_bn(out).transpose(1, 2)

        out += identity
        out = self.act(out)
        # out = self.relu(out)

        return out


class TDCBAM(nn.Module):
    # input should be like [Batch, time, frequency]
    def __init__(self, inplanes, planes, time_freq='time', pooling='avg'):
        super(TDCBAM, self).__init__()
        self.time_freq = time_freq
        self.activation = nn.Sigmoid()
        self.pooling = pooling

        self.cov_t = nn.Conv2d(1, 1, kernel_size=(7, 1), stride=1, padding=(3, 0))
        self.avg_t = nn.AdaptiveAvgPool2d((None, 1))
        self.max_t = nn.AdaptiveMaxPool2d((None, 1))

        self.cov_f = nn.Conv2d(1, 1, kernel_size=(1, 7), stride=1, padding=(0, 3))
        self.avg_f = nn.AdaptiveAvgPool2d((1, None))
        self.max_f = nn.AdaptiveMaxPool2d((1, None))

    def forward(self, input):
        if len(input.shape) == 3:
            input = input.unsqueeze(1)

        t_output = self.avg_t(input)
        if self.pooling == 'both':
            t_output += self.max_t(input)

        t_output = self.cov_t(t_output)
        t_output = self.activation(t_output)
        t_output = input * t_output

        f_output = self.avg_f(input)
        if self.pooling == 'both':
            f_output += self.max_f(input)

        f_output = self.cov_f(f_output)
        f_output = self.activation(f_output)
        f_output = input * f_output

        output = (t_output + f_output) / 2

        if len(input.shape) == 4:
            output = output.squeeze(1)

        return output


class TDNNCBAMBlock(nn.Module):

    def __init__(self, inplanes, planes, downsample=None, dilation=1, **kwargs):
        super(TDNNCBAMBlock, self).__init__()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        if isinstance(downsample, int) and downsample > 0:
            inter_connect = int(planes / downsample)
        else:
            inter_connect = planes

        self.tdnn1 = TimeDelayLayer_v5(input_dim=inplanes, output_dim=inter_connect, context_size=3,
                                       stride=1, dilation=dilation, padding=1)
        # self.relu = nn.ReLU(inplace=True)

        self.tdnn2 = TimeDelayLayer_v5(input_dim=inter_connect, output_dim=planes, context_size=3,
                                       stride=1, dilation=dilation, padding=1)

        self.CBAM_layer = TDCBAM(planes, planes)
        # self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.tdnn1(x)
        out = self.tdnn2(out)

        out = self.CBAM_layer(out)

        out += identity
        # out = self.relu(out)

        return out


class TDNNCBAMBlock_v2(nn.Module):

    def __init__(self, inplanes, planes, downsample=None, dilation=1, activation='relu', **kwargs):
        super(TDNNCBAMBlock_v2, self).__init__()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        if isinstance(downsample, int) and downsample > 0:
            inter_connect = int(planes / downsample)
        else:
            inter_connect = planes

        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation in ['leakyrelu', 'leaky_relu']:
            act_fn = nn.LeakyReLU
        elif activation == 'prelu':
            act_fn = nn.PReLU

        self.tdnn1_kernel = nn.Conv1d(inplanes, inter_connect, 3, stride=1,
                                padding=1, dilation=dilation, bias=False)
        self.tdnn1_bn = nn.BatchNorm1d(inter_connect)
        self.act = act_fn()

        self.tdnn2_kernel = nn.Conv1d(inter_connect, planes, 3, stride=1,
                                      padding=1, dilation=dilation, bias=False)
        self.tdnn2_bn = nn.BatchNorm1d(planes)

        self.CBAM_layer = TDCBAM(planes, planes)

    def forward(self, x):
        identity = x

        out = self.tdnn1_kernel(x.transpose(1, 2))
        out = self.tdnn1_bn(out)
        out = self.act(out)

        out = self.tdnn2_kernel(out)
        out = self.tdnn2_bn(out)

        out = self.CBAM_layer(out).transpose(1, 2)
        out += identity
        out = self.act(out)

        return out


class SqueezeExcitation(nn.Module):
    # input should be like [Batch, channel, time, frequency]
    def __init__(self, inplanes, reduction_ratio=2):
        super(SqueezeExcitation, self).__init__()
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(inplanes, max(int(inplanes / self.reduction_ratio), 1))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(max(int(inplanes / self.reduction_ratio), 1), inplanes)
        self.activation = nn.Sigmoid()

    def forward(self, input):
        scale = input.mean(dim=2)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.activation(scale).unsqueeze(2)

        output = input * scale

        return output

    def __repr__(self):
        return "SqueezeExcitation(reduction_ratio=%f)" % self.reduction_ratio


class TDNNSEBlock_v2(nn.Module):

    def __init__(self, inplanes, planes, downsample=None, dilation=1,
                 activation='relu', reduction_ratio=2, **kwargs):
        super(TDNNSEBlock_v2, self).__init__()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        if isinstance(downsample, int) and downsample > 0:
            inter_connect = int(planes / downsample)
        else:
            inter_connect = planes

        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation in ['leakyrelu', 'leaky_relu']:
            act_fn = nn.LeakyReLU
        elif activation == 'prelu':
            act_fn = nn.PReLU

        self.tdnn1_kernel = nn.Conv1d(inplanes, inter_connect, 3, stride=1,
                                      padding=1, dilation=dilation, bias=False)
        self.tdnn1_bn = nn.BatchNorm1d(inter_connect)
        self.act = act_fn()
        self.reduction_ratio = reduction_ratio

        self.tdnn2_kernel = nn.Conv1d(inter_connect, planes, 3, stride=1,
                                      padding=1, dilation=dilation, bias=False)
        self.tdnn2_bn = nn.BatchNorm1d(planes)

        self.SE_layer = SqueezeExcitation(inplanes=planes, reduction_ratio=reduction_ratio)

    def forward(self, x):
        identity = x

        out = self.tdnn1_kernel(x.transpose(1, 2))
        out = self.tdnn1_bn(out)
        out = self.act(out)

        out = self.tdnn2_kernel(out)
        out = self.tdnn2_bn(out)

        out = self.SE_layer(out).transpose(1, 2)

        out += identity
        out = self.act(out)

        return out


class TDNNBlock_v6(nn.Module):

    def __init__(self, inplanes, planes, downsample=None, dilation=1, **kwargs):
        super(TDNNBlock_v6, self).__init__()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        if isinstance(downsample, int):
            inter_connect = int(planes / downsample)
        else:
            inter_connect = planes

        self.tdnn1 = TimeDelayLayer_v6(input_dim=inplanes, output_dim=inter_connect, context_size=3,
                                       stride=1, dilation=dilation, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.tdnn2 = TimeDelayLayer_v6(input_dim=inter_connect, output_dim=planes, context_size=3,
                                       stride=1, dilation=dilation, padding=1)
        # self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.tdnn1(x)
        out = self.tdnn2(out)

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
        # self.relu = nn.ReLU(inplace=True)

        self.tdnn2 = TimeDelayLayer_v5(input_dim=inplanes, output_dim=inplanes * 2, context_size=3,
                                       stride=1, dilation=dilation, padding=1, groups=groups)

        self.tdnn3 = TimeDelayLayer_v5(input_dim=inplanes * 2, output_dim=planes, context_size=1,
                                       stride=1, dilation=dilation)

        # self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.tdnn1(x)
        out = self.tdnn2(out)
        out = self.tdnn3(out)

        # if self.downsample is not None:
        #     identity = self.downsample(x)

        out += identity
        # out = self.relu(out)

        return out


class TDNNBottleBlock_v2(nn.Module):

    def __init__(self, inplanes, planes, downsample=None, dilation=1, activation='relu', **kwargs):
        super(TDNNBottleBlock_v2, self).__init__()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        if isinstance(downsample, int):
            inter_connect = int(planes / downsample)
        else:
            inter_connect = planes

        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation in ['leakyrelu', 'leaky_relu']:
            act_fn = nn.LeakyReLU
        elif activation == 'prelu':
            act_fn = nn.PReLU

        self.tdnn1_kernel = nn.Conv1d(inplanes, inter_connect, 1, stride=1,
                                      padding=0, dilation=dilation, bias=False)
        self.tdnn1_bn = nn.BatchNorm1d(inter_connect)
        self.act = act_fn()

        self.tdnn2_kernel = nn.Conv1d(inter_connect, planes, 3, stride=1,
                                      padding=1, dilation=dilation, bias=False)
        self.tdnn2_bn = nn.BatchNorm1d(planes)

        self.tdnn3_kernel = nn.Conv1d(inter_connect, planes, 1, stride=1,
                                      padding=0, dilation=dilation, bias=False)
        self.tdnn3_bn = nn.BatchNorm1d(planes)

        # self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.tdnn1_kernel(x.transpose(1, 2))
        out = self.tdnn1_bn(out)
        out = self.act(out)

        out = self.tdnn2_kernel(out)
        out = self.tdnn2_bn(out)  # .transpose(1, 2)
        out = self.act(out)

        out = self.tdnn3_kernel(out)
        out = self.tdnn3_bn(out).transpose(1, 2)

        out += identity
        out = self.act(out)
        # out = self.relu(out)

        return out


class ShuffleTDNNBlock(nn.Module):

    def __init__(self, inplanes=512, planes=512, context_size=5, stride=1, dilation=1,
                 dropout_p=0.0, padding=0, groups=1, activation='relu') -> None:
        super(ShuffleTDNNBlock, self).__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = inplanes
        self.output_dim = planes
        self.dilation = dilation
        self.dropout_p = dropout_p
        self.padding = padding
        self.groups = groups
        self.activation = activation
        nonlinearity = get_activation(activation)

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = self.output_dim // 2
        assert (self.stride != 1) or (self.input_dim == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(self.input_dim, self.input_dim, kernel_size=3, stride=self.stride, padding=1,
                                    dilation=dilation),
                nn.BatchNorm1d(self.input_dim),
                nn.Conv1d(self.input_dim, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm1d(branch_features),
                nonlinearity(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv1d(self.input_dim if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(branch_features),
            nonlinearity(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=self.context_size,
                                dilation=self.dilation, stride=self.stride, padding=int((self.context_size - 1) / 2)),
            nn.BatchNorm1d(branch_features),
            nn.Conv1d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(branch_features),
            nonlinearity(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i: int, o: int, kernel_size: int, stride: int = 1, padding: int = 0,
                       dilation: int = 1, bias: bool = False) -> nn.Conv1d:
        return nn.Conv1d(i, o, kernel_size, stride, padding, dilation=dilation, bias=bias, groups=i)

    def forward(self, x):
        x = x.transpose(1, 2)
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out.transpose(1, 2)


class RET(nn.Module):
    def __init__(self, num_classes, embedding_size, input_dim, alpha=0., input_norm='',
                 channels=[512, 512, 512, 512, 512, 1536], context=[5, 3, 3, 5], activation='relu',
                 downsample=None, resnet_size=17, dilation=[1, 1, 1, 1], stride=[1],
                 red_ratio=2,
                 dropout_p=0.0, dropout_layer=False, encoder_type='STAP', block_type='basic',
                 mask='None', mask_len=20, **kwargs):
        super(RET, self).__init__()
        self.num_classes = num_classes
        self.dropout_p = dropout_p
        self.dropout_layer = dropout_layer
        self.input_dim = input_dim
        self.alpha = alpha
        self.mask = mask
        self.channels = channels
        self.context = context
        self.activation = activation
        self.red_ratio = red_ratio

        tdnn_type = {14: [1, 1, 1, 0],
                     17: [1, 1, 1, 1],
                     21: [1, 1, 1, 1],
                     34: [3, 4, 6, 3], }
        self.layers = tdnn_type[resnet_size] if resnet_size in tdnn_type else tdnn_type[17]
        self.stride = stride
        if len(self.stride) == 1:
            while len(self.stride) < 4:
                self.stride.append(self.stride[0])

        if input_norm == 'Inst':
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

        TDNN_layer = TimeDelayLayer_v5
        if block_type.lower() == 'basic':
            Blocks = TDNNBlock
        elif block_type.lower() == 'basic_v2':
            Blocks = TDNNBlock_v2
        elif block_type.lower() == 'basic_v6':
            Blocks = TDNNBlock_v6
            TDNN_layer = TimeDelayLayer_v6
        elif block_type.lower() == 'shuffle':
            Blocks = TDNNBlock
            TDNN_layer = ShuffleTDLayer
        elif block_type.lower() == 'shublock':
            Blocks = ShuffleTDNNBlock
        elif block_type.lower() == 'agg':
            Blocks = TDNNBottleBlock
        elif block_type.lower() == 'agg_v2':
            Blocks = TDNNBottleBlock_v2
        elif block_type.lower() == 'cbam':
            Blocks = TDNNCBAMBlock
        elif block_type.lower() == 'cbam_v2':
            TDNN_layer = TimeDelayLayer_v6
            Blocks = TDNNCBAMBlock_v2
        elif block_type.lower() == 'seblock_v2':
            Blocks = TDNNSEBlock_v2
        else:
            raise ValueError(block_type)

        self.frame1 = TDNN_layer(input_dim=self.input_dim, output_dim=self.channels[0],
                                 context_size=self.context[0], dilation=dilation[0], stride=self.stride[0],
                                 activation=self.activation)

        self.frame2 = Blocks(inplanes=self.channels[0], planes=self.channels[0],
                             downsample=downsample, dilation=1, activation=self.activation)

        self.frame4 = TDNN_layer(input_dim=self.channels[0], output_dim=self.channels[1],
                                 context_size=self.context[1], dilation=dilation[1], stride=self.stride[1],
                                 activation=self.activation)
        self.frame5 = Blocks(inplanes=self.channels[1], planes=self.channels[1],
                             downsample=downsample, dilation=1, activation=self.activation)

        self.frame7 = TDNN_layer(input_dim=self.channels[1], output_dim=self.channels[2],
                                 context_size=self.context[2], dilation=dilation[2], stride=self.stride[2],
                                 activation=self.activation)
        self.frame8 = Blocks(inplanes=self.channels[2], planes=self.channels[2],
                             downsample=downsample, dilation=1, activation=self.activation)

        if self.layers[3] != 0:
            self.frame10 = TDNN_layer(input_dim=self.channels[2], output_dim=self.channels[3],
                                      context_size=self.context[3], dilation=dilation[3], stride=self.stride[3],
                                      activation=self.activation)
            self.frame11 = Blocks(inplanes=self.channels[3], planes=self.channels[3],
                                  downsample=downsample, dilation=1, activation=self.activation)

        self.frame13 = TDNN_layer(input_dim=self.channels[3], output_dim=self.channels[4],
                                  context_size=1, dilation=1, activation=self.activation)
        self.frame14 = TDNN_layer(input_dim=self.channels[4], output_dim=self.channels[5],
                                  context_size=1, dilation=1, activation=self.activation)

        self.drop = nn.Dropout(p=self.dropout_p)

        if encoder_type == 'STAP':
            self.encoder = StatisticPooling(input_dim=self.channels[5])
        elif encoder_type in ['SASP', 'ASTP']:
            self.encoder = AttentionStatisticPooling(input_dim=self.channels[5], hidden_dim=int(embedding_size / 2))
        else:
            raise ValueError(encoder_type)

        if activation == 'relu':
            nonlinearity = nn.ReLU
        elif activation in ['leakyrelu', 'leaky_relu']:
            nonlinearity = nn.LeakyReLU
        elif activation == 'prelu':
            nonlinearity = nn.PReLU

        self.segment1 = nn.Sequential(
            nn.Linear(self.channels[5] * 2, 512),
            nonlinearity(),
            nn.BatchNorm1d(512)
        )

        self.segment2 = nn.Sequential(
            nn.Linear(512, embedding_size),
            nonlinearity(),
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
                nonlinear = 'leaky_relu' if self.activation == 'leakyrelu' else self.activation
                nn.init.kaiming_normal_(m.kernel.weight, mode='fan_out', nonlinearity=nonlinear)

    def forward(self, x):
        # pdb.set_trace()
        if len(x.shape) == 4:
            x = x.squeeze(1).float()

        if self.inst_layer != None:
            x = self.inst_layer(x)

        if self.mask_layer != None:
            x = self.mask_layer(x)

        # x = x.transpose(1, 2)
        x = self.frame1(x)
        x = self.frame2(x)

        x = self.frame4(x)
        x = self.frame5(x)

        x = self.frame7(x)
        x = self.frame8(x)

        if self.layers[3] != 0:
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


class RET_v2(nn.Module):
    def __init__(self, num_classes, embedding_size, input_dim, alpha=0., input_norm='',
                 channels=[512, 512, 512, 512, 512, 1536], context=[5, 3, 3, 5],
                 downsample=None, resnet_size=17, stride=[1], activation='relu',
                 dilation=[1, 1, 1, 1],
                 dropout_p=0.0, dropout_layer=False, encoder_type='STAP', block_type='Basic',
                 mask='None', mask_len=20, **kwargs):
        super(RET_v2, self).__init__()
        self.num_classes = num_classes
        self.dropout_p = dropout_p
        self.dropout_layer = dropout_layer
        self.input_dim = input_dim
        self.alpha = alpha
        self.mask = mask
        self.channels = channels
        self.context = context
        self.stride = stride
        self.activation = activation
        self.dilation = dilation

        if len(self.stride) == 1:
            while len(self.stride) < 4:
                self.stride.append(self.stride[0])

        self.tdnn_size = resnet_size
        tdnn_type = {14: [1, 1, 1, 0],
                     17: [1, 1, 1, 1],
                     18: [2, 2, 2, 2]}
        self.layers = tdnn_type[resnet_size] if resnet_size in tdnn_type else tdnn_type[17]

        if input_norm == 'Inst':
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

        TDNN_layer = TimeDelayLayer_v5
        if block_type.lower() == 'basic':
            Blocks = TDNNBlock
        elif block_type.lower() == 'basic_v2':
            Blocks = TDNNBlock_v2
        elif block_type.lower() == 'basic_v6':
            Blocks = TDNNBlock_v6
            TDNN_layer = TimeDelayLayer_v6
        elif block_type.lower() == 'shuffle':
            Blocks = TDNNBlock
            TDNN_layer = ShuffleTDLayer
        elif block_type.lower() == 'agg':
            Blocks = TDNNBottleBlock
        elif block_type.lower() == 'cbam':
            Blocks = TDNNCBAMBlock
        elif block_type.lower() == 'cbam_v2':
            TDNN_layer = TimeDelayLayer_v6
            Blocks = TDNNCBAMBlock_v2
        else:
            raise ValueError(block_type)

        self.frame1 = TDNN_layer(input_dim=self.input_dim, output_dim=self.channels[0],
                                 context_size=self.context[0], dilation=self.dilation[0], stride=self.stride[0],
                                 activation=self.activation)
        self.frame2 = self._make_block(block=Blocks, inplanes=self.channels[0], planes=self.channels[0],
                                       downsample=downsample, dilation=1, blocks=self.layers[0],
                                       activation=self.activation)

        self.frame4 = TDNN_layer(input_dim=self.channels[0], output_dim=self.channels[1],
                                 context_size=self.context[1], dilation=self.dilation[1], stride=self.stride[1],
                                 activation=self.activation)
        self.frame5 = self._make_block(block=Blocks, inplanes=self.channels[1], planes=self.channels[1],
                                       downsample=downsample, dilation=1, blocks=self.layers[1],
                                       activation=self.activation)

        self.frame7 = TDNN_layer(input_dim=self.channels[1], output_dim=self.channels[2],
                                 context_size=self.context[2], dilation=self.dilation[2], stride=self.stride[2],
                                 activation=self.activation)

        self.frame8 = self._make_block(block=Blocks, inplanes=self.channels[2], planes=self.channels[2],
                                       downsample=downsample, dilation=1, blocks=self.layers[2],
                                       activation=self.activation)

        if self.layers[3] != 0:
            self.frame10 = TDNN_layer(input_dim=self.channels[2], output_dim=self.channels[3],
                                      context_size=self.context[3], dilation=self.dilation[3], stride=self.stride[3],
                                      activation=self.activation)
            self.frame11 = self._make_block(block=Blocks, inplanes=self.channels[3], planes=self.channels[3],
                                            downsample=downsample, dilation=1, blocks=self.layers[3],
                                            activation=self.activation)

        self.frame13 = TDNN_layer(input_dim=self.channels[3], output_dim=self.channels[4],
                                  context_size=1, dilation=1, activation=self.activation)
        self.frame14 = TDNN_layer(input_dim=self.channels[4], output_dim=self.channels[5],
                                  context_size=1, dilation=1, activation=self.activation)

        self.drop = nn.Dropout(p=self.dropout_p)

        if encoder_type == 'STAP':
            self.encoder = StatisticPooling(input_dim=self.channels[5])
        elif encoder_type == 'SASP':
            self.encoder = AttentionStatisticPooling(input_dim=self.channels[5], hidden_dim=512)
        else:
            raise ValueError(encoder_type)

        self.segment1 = nn.Sequential(
            nn.Linear(self.channels[5] * 2, 512),
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
                nn.init.kaiming_normal_(m.kernel.weight, mode='fan_out', nonlinearity=self.activation)

    def _make_block(self, block, inplanes, planes, downsample, dilation, activation, blocks=1):
        if blocks == 0:
            return None
        layers = []
        layers.append(block(inplanes=inplanes, planes=planes, downsample=downsample,
                            dilation=dilation, activation=activation))
        for _ in range(1, blocks):
            layers.append(block(inplanes=inplanes, planes=planes, downsample=downsample,
                                dilation=dilation, activation=activation))

        return nn.Sequential(*layers)

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

        if self.layers[3] != 0:
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


class RET_v3(nn.Module):
    def __init__(self, num_classes, embedding_size, input_dim, alpha=0., input_norm='',
                 channels=[512, 512, 512, 512, 512, 1536], context=[5, 3, 3, 5],
                 downsample=None, resnet_size=17, stride=[1], activation='relu',
                 dilation=[1, 1, 1, 1], red_ratio=2,
                 dropout_p=0.0, dropout_layer=False, encoder_type='STAP', block_type='Basic',
                 mask='None', mask_len=20, **kwargs):
        super(RET_v3, self).__init__()
        self.num_classes = num_classes
        self.dropout_p = dropout_p
        self.dropout_layer = dropout_layer
        self.input_dim = input_dim
        self.alpha = alpha
        self.mask = mask
        self.channels = channels
        self.context = context
        self.stride = stride
        self.activation = activation
        self.dilation = dilation
        self.red_ratio = red_ratio

        if len(self.stride) == 1:
            while len(self.stride) < 4:
                self.stride.append(self.stride[0])

        self.tdnn_size = resnet_size
        tdnn_type = {14: [1, 1, 1, 0],
                     17: [1, 1, 1, 1],
                     18: [2, 2, 2, 2]}
        self.layers = tdnn_type[resnet_size] if resnet_size in tdnn_type else tdnn_type[17]

        if input_norm == 'Inst':
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

        TDNN_layer = TimeDelayLayer_v5
        if block_type.lower() == 'basic':
            Blocks = TDNNBlock
        elif block_type.lower() == 'basic_v2':
            Blocks = TDNNBlock_v2
        elif block_type.lower() == 'basic_v6':
            Blocks = TDNNBlock_v6
            TDNN_layer = TimeDelayLayer_v6
        elif block_type.lower() == 'shuffle':
            Blocks = TDNNBlock
            TDNN_layer = ShuffleTDLayer
        elif block_type.lower() == 'shublock':
            Blocks = ShuffleTDNNBlock
        elif block_type.lower() == 'agg':
            Blocks = TDNNBottleBlock
        elif block_type.lower() == 'cbam':
            Blocks = TDNNCBAMBlock
        elif block_type.lower() == 'cbam_v2':
            TDNN_layer = TimeDelayLayer_v6
            Blocks = TDNNCBAMBlock_v2
        elif block_type.lower() == 'seblock_v2':
            Blocks = TDNNSEBlock_v2
        else:
            raise ValueError(block_type)

        self.frame1 = TDNN_layer(input_dim=self.input_dim, output_dim=self.channels[0],
                                 context_size=self.context[0], dilation=self.dilation[0], stride=self.stride[0],
                                 activation=self.activation)
        self.frame2 = self._make_block(block=Blocks, inplanes=self.channels[0], planes=self.channels[0],
                                       downsample=downsample, dilation=1, blocks=self.layers[0],
                                       activation=self.activation, reduction_ratio=self.red_ratio)

        self.frame4 = TDNN_layer(input_dim=self.channels[0], output_dim=self.channels[1],
                                 context_size=self.context[1], dilation=self.dilation[1], stride=self.stride[1],
                                 activation=self.activation)
        self.frame5 = self._make_block(block=Blocks, inplanes=self.channels[1], planes=self.channels[1],
                                       downsample=downsample, dilation=1, blocks=self.layers[1],
                                       activation=self.activation, reduction_ratio=self.red_ratio)

        self.frame7 = TDNN_layer(input_dim=self.channels[1], output_dim=self.channels[2],
                                 context_size=self.context[2], dilation=self.dilation[2], stride=self.stride[2],
                                 activation=self.activation)

        self.frame8 = self._make_block(block=Blocks, inplanes=self.channels[2], planes=self.channels[2],
                                       downsample=downsample, dilation=1, blocks=self.layers[2],
                                       activation=self.activation, reduction_ratio=self.red_ratio)

        if self.layers[3] != 0:
            self.frame10 = TDNN_layer(input_dim=self.channels[2], output_dim=self.channels[3],
                                      context_size=self.context[3], dilation=self.dilation[3], stride=self.stride[3],
                                      activation=self.activation)
            self.frame11 = self._make_block(block=Blocks, inplanes=self.channels[3], planes=self.channels[3],
                                            downsample=downsample, dilation=1, blocks=self.layers[3],
                                            activation=self.activation, reduction_ratio=self.red_ratio)

        self.frame13 = TDNN_layer(input_dim=self.channels[3], output_dim=self.channels[4],
                                  context_size=1, dilation=1, activation=self.activation)
        self.frame14 = TDNN_layer(input_dim=self.channels[4], output_dim=self.channels[5],
                                  context_size=1, dilation=1, activation=self.activation)

        self.drop = nn.Dropout(p=self.dropout_p)

        if encoder_type == 'STAP':
            self.encoder = StatisticPooling(input_dim=self.channels[5])
        elif encoder_type in ['SASP', 'ASTP']:
            self.encoder = AttentionStatisticPooling(input_dim=self.channels[5], hidden_dim=int(embedding_size / 2))
        else:
            raise ValueError(encoder_type)

        if activation == 'relu':
            nonlinearity = nn.ReLU
        elif activation in ['leakyrelu', 'leaky_relu']:
            nonlinearity = nn.LeakyReLU
        elif activation == 'prelu':
            nonlinearity = nn.PReLU

        self.segment1 = nn.Sequential(
            nn.Linear(self.channels[5] * 2, embedding_size),
            nonlinearity(),
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
                nonlinear = 'leaky_relu' if self.activation == 'leakyrelu' else self.activation
                nn.init.kaiming_normal_(m.kernel.weight, mode='fan_out', nonlinearity=nonlinear)

    def _make_block(self, block, inplanes, planes, downsample, dilation, activation, blocks=1, reduction_ratio=2):
        if blocks == 0:
            return None
        layers = []
        layers.append(block(inplanes=inplanes, planes=planes, downsample=downsample,
                            dilation=dilation, activation=activation, reduction_ratio=reduction_ratio))
        for _ in range(1, blocks):
            layers.append(block(inplanes=inplanes, planes=planes, downsample=downsample,
                                dilation=dilation, activation=activation, reduction_ratio=reduction_ratio))

        return nn.Sequential(*layers)

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

        if self.layers[3] != 0:
            x = self.frame10(x)
            x = self.frame11(x)

        x = self.frame13(x)
        x = self.frame14(x)

        if self.dropout_layer:
            x = self.drop(x)

        # print(x.shape)
        x = self.encoder(x)
        embedding_a = self.segment1(x)

        if self.alpha:
            embedding_a = self.l2_norm(embedding_a)

        logits = self.classifier(embedding_a)

        return logits, embedding_a
