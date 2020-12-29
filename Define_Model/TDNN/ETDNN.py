#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: ETDNN.py
@Time: 2020/12/29 12:04
@Overview:
"""
import torch.nn as nn

from Define_Model.FilterLayer import Mean_Norm
from Define_Model.Pooling import StatisticPooling
from Define_Model.TDNN.TDNN import TimeDelayLayer_v2, TimeDelayLayer_v4, TimeDelayLayer_v5


class ETDNN(nn.Module):
    def __init__(self, num_classes, embedding_size=256, batch_norm=True, input_norm='Mean',
                 input_dim=80, dropout_p=0.0, encoder_type='STAP', **kwargs):
        super(ETDNN, self).__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim

        if input_norm == 'Instance':
            self.inst_layer = nn.InstanceNorm1d(input_dim)
        elif input_norm == 'Mean':
            self.inst_layer = Mean_Norm()
        else:
            self.inst_layer = None

        self.dropout_p = dropout_p

        self.frame1 = TimeDelayLayer_v2(input_dim=input_dim, output_dim=512, context_size=5, dilation=1,
                                        activation='leakyrelu', batch_norm=batch_norm, dropout_p=dropout_p)
        self.affine2 = TimeDelayLayer_v2(input_dim=512, output_dim=512, context_size=1, dilation=1,
                                         activation='leakyrelu', batch_norm=batch_norm, dropout_p=dropout_p)
        self.frame3 = TimeDelayLayer_v2(input_dim=512, output_dim=512, context_size=3, dilation=2,
                                        activation='leakyrelu', batch_norm=batch_norm, dropout_p=dropout_p)
        self.affine4 = TimeDelayLayer_v2(input_dim=512, output_dim=512, context_size=1, dilation=1,
                                         activation='leakyrelu', batch_norm=batch_norm, dropout_p=dropout_p)
        self.frame5 = TimeDelayLayer_v2(input_dim=512, output_dim=512, context_size=3, dilation=3,
                                        activation='leakyrelu', batch_norm=batch_norm, dropout_p=dropout_p)
        self.affine6 = TimeDelayLayer_v2(input_dim=512, output_dim=512, context_size=1, dilation=1,
                                         activation='leakyrelu', batch_norm=batch_norm, dropout_p=dropout_p)
        self.frame7 = TimeDelayLayer_v2(input_dim=512, output_dim=512, context_size=3, dilation=4,
                                        activation='leakyrelu', batch_norm=batch_norm, dropout_p=dropout_p)
        self.frame8 = TimeDelayLayer_v2(input_dim=512, output_dim=512, context_size=1, dilation=1,
                                        activation='leakyrelu', batch_norm=batch_norm, dropout_p=dropout_p)
        self.frame9 = TimeDelayLayer_v2(input_dim=512, output_dim=1500, context_size=1, dilation=1,
                                        activation='leakyrelu', batch_norm=batch_norm, dropout_p=dropout_p)

        # self.segment11 = nn.Linear(3000, embedding_size)
        # self.leakyrelu = nn.LeakyReLU()
        # self.batchnorm = nn.BatchNorm1d(embedding_size)

        if encoder_type == 'STAP':
            self.encoder = StatisticPooling(input_dim=1500)
        else:
            self.encoder = nn.AdaptiveAvgPool2d((1, None))

        self.segment11 = nn.Sequential(nn.Linear(3000, embedding_size),
                                       nn.LeakyReLU(),
                                       nn.BatchNorm1d(embedding_size))

        self.classifier = nn.Linear(embedding_size, num_classes)

        for m in self.modules():  # 对于各层参数的初始化
            if isinstance(m, nn.BatchNorm1d):  # weight设置为1，bias为0
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, TimeDelayLayer_v2):
                nn.init.kaiming_normal_(m.kernel.weight, mode='fan_out', nonlinearity='leaky_relu')

    # def statistic_pooling(self, x):
    #     mean_x = x.mean(dim=1)
    #     # std_x = x.std(dim=1)
    #     std_x = x.var(dim=1, unbiased=False).add_(1e-12).sqrt()
    #     mean_std = torch.cat((mean_x, std_x), 1)
    #     return mean_std

    def set_global_dropout(self, dropout_p):
        self.dropout_p = dropout_p

        for m in self.modules():
            if isinstance(m, TimeDelayLayer_v2):
                m.set_dropout(dropout_p)

    def forward(self, x):
        # pdb.set_trace()
        if x.shape[1] == 1:
            x = x.squeeze(1).float()

        if self.inst_layer != None:
            x = self.inst_layer(x)

        x = self.frame1(x)
        x = self.affine2(x)
        x = self.frame3(x)
        x = self.affine4(x)
        x = self.frame5(x)
        x = self.affine6(x)
        x = self.frame7(x)
        x = self.frame8(x)
        x = self.frame9(x)

        x = self.encoder(x)
        embeddings = self.segment11(x)

        logits = self.classifier(embeddings)

        return logits, embeddings


class ETDNN_v4(nn.Module):
    def __init__(self, num_classes, embedding_size=256, batch_norm=True, input_norm='Mean',
                 input_dim=80, dropout_p=0.0, encoder_type='STAP', activation='leakyrelu', **kwargs):
        super(ETDNN_v4, self).__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim

        if input_norm == 'Instance':
            self.inst_layer = nn.InstanceNorm1d(input_dim)
        elif input_norm == 'Mean':
            self.inst_layer = Mean_Norm()
        else:
            self.inst_layer = None

        self.dropout_p = dropout_p

        self.frame1 = TimeDelayLayer_v4(input_dim=input_dim, output_dim=512, context_size=5, dilation=1,
                                        activation=activation, batch_norm=batch_norm, dropout_p=dropout_p)
        self.affine2 = TimeDelayLayer_v4(input_dim=512, output_dim=512, context_size=1, dilation=1,
                                         activation=activation, batch_norm=batch_norm, dropout_p=dropout_p)
        self.frame3 = TimeDelayLayer_v4(input_dim=512, output_dim=512, context_size=3, dilation=2,
                                        activation=activation, batch_norm=batch_norm, dropout_p=dropout_p)
        self.affine4 = TimeDelayLayer_v4(input_dim=512, output_dim=512, context_size=1, dilation=1,
                                         activation=activation, batch_norm=batch_norm, dropout_p=dropout_p)
        self.frame5 = TimeDelayLayer_v4(input_dim=512, output_dim=512, context_size=3, dilation=3,
                                        activation=activation, batch_norm=batch_norm, dropout_p=dropout_p)
        self.affine6 = TimeDelayLayer_v4(input_dim=512, output_dim=512, context_size=1, dilation=1,
                                         activation=activation, batch_norm=batch_norm, dropout_p=dropout_p)
        self.frame7 = TimeDelayLayer_v4(input_dim=512, output_dim=512, context_size=3, dilation=4,
                                        activation=activation, batch_norm=batch_norm, dropout_p=dropout_p)
        self.frame8 = TimeDelayLayer_v4(input_dim=512, output_dim=512, context_size=1, dilation=1,
                                        activation=activation, batch_norm=batch_norm, dropout_p=dropout_p)
        self.frame9 = TimeDelayLayer_v4(input_dim=512, output_dim=512, context_size=1, dilation=1,
                                        activation=activation, batch_norm=batch_norm, dropout_p=dropout_p)
        self.frame10 = TimeDelayLayer_v4(input_dim=512, output_dim=1500, context_size=1, dilation=1,
                                         activation=activation, batch_norm=batch_norm, dropout_p=dropout_p)

        # self.segment11 = nn.Linear(3000, embedding_size)
        # self.leakyrelu = nn.LeakyReLU()
        # self.batchnorm = nn.BatchNorm1d(embedding_size)

        if encoder_type == 'STAP':
            self.encoder = StatisticPooling(input_dim=1500)
        else:
            self.encoder = nn.AdaptiveAvgPool2d((1, None))

        seg12 = [nn.Linear(3000, embedding_size)]
        seg13 = [nn.Linear(embedding_size, embedding_size)]
        if activation == 'relu':
            seg12.append(nn.ReLU())
            seg13.append(nn.ReLU())
        elif activation == 'leakyrelu':
            seg12.append(nn.LeakyReLU())
            seg13.append(nn.LeakyReLU())
        elif activation == 'prelu':
            seg12.append(nn.PReLU())
            seg13.append(nn.PReLU())
        seg12.append(nn.BatchNorm1d(embedding_size))
        seg13.append(nn.BatchNorm1d(embedding_size))

        self.segment12 = nn.Sequential(seg12)
        self.segment13 = nn.Sequential(seg13)

        self.classifier = nn.Linear(embedding_size, num_classes)

        for m in self.modules():  # 对于各层参数的初始化
            if isinstance(m, nn.BatchNorm1d):  # weight设置为1，bias为0
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def set_global_dropout(self, dropout_p):
        self.dropout_p = dropout_p

        for m in self.modules():
            if isinstance(m, ETDNN_v4):
                m.set_dropout(dropout_p)

    def forward(self, x):
        # pdb.set_trace()
        if self.inst_layer != None:
            x = self.inst_layer(x)

        x = self.frame1(x)
        x = self.affine2(x)
        x = self.frame3(x)
        x = self.affine4(x)
        x = self.frame5(x)
        x = self.affine6(x)
        x = self.frame7(x)
        x = self.frame8(x)
        x = self.frame9(x)
        x = self.frame10(x)

        x = self.encoder(x)
        embeddings_a = self.segment12(x)
        embeddings_b = self.segment13(embeddings_a)

        logits = self.classifier(embeddings_b)

        return logits, embeddings_b

    def xvector(self, x):

        if self.inst_layer != None:
            x = self.inst_layer(x)

        x = self.frame1(x)
        x = self.affine2(x)
        x = self.frame3(x)
        x = self.affine4(x)
        x = self.frame5(x)
        x = self.affine6(x)
        x = self.frame7(x)
        x = self.frame8(x)
        x = self.frame9(x)
        x = self.frame10(x)

        x = self.encoder(x)
        embeddings_a = self.segment12(x)

        return embeddings_a


class ETDNN_v5(nn.Module):
    def __init__(self, num_classes, embedding_size=256, batch_norm=True, input_norm='Mean',
                 input_dim=80, dropout_p=0.0, encoder_type='STAP', activation='relu', **kwargs):
        super(ETDNN_v5, self).__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim

        if input_norm == 'Instance':
            self.inst_layer = nn.InstanceNorm1d(input_dim)
        elif input_norm == 'Mean':
            self.inst_layer = Mean_Norm()
        else:
            self.inst_layer = None

        self.dropout_p = dropout_p

        self.frame1 = TimeDelayLayer_v5(input_dim=input_dim, output_dim=512, context_size=5, dilation=1,
                                        activation=activation, batch_norm=batch_norm, dropout_p=dropout_p)
        self.affine2 = TimeDelayLayer_v5(input_dim=512, output_dim=512, context_size=1, dilation=1,
                                         activation=activation, batch_norm=batch_norm, dropout_p=dropout_p)
        self.frame3 = TimeDelayLayer_v5(input_dim=512, output_dim=512, context_size=3, dilation=2,
                                        activation=activation, batch_norm=batch_norm, dropout_p=dropout_p)
        self.affine4 = TimeDelayLayer_v5(input_dim=512, output_dim=512, context_size=1, dilation=1,
                                         activation=activation, batch_norm=batch_norm, dropout_p=dropout_p)
        self.frame5 = TimeDelayLayer_v5(input_dim=512, output_dim=512, context_size=3, dilation=3,
                                        activation=activation, batch_norm=batch_norm, dropout_p=dropout_p)
        self.affine6 = TimeDelayLayer_v5(input_dim=512, output_dim=512, context_size=1, dilation=1,
                                         activation=activation, batch_norm=batch_norm, dropout_p=dropout_p)
        self.frame7 = TimeDelayLayer_v5(input_dim=512, output_dim=512, context_size=3, dilation=4,
                                        activation=activation, batch_norm=batch_norm, dropout_p=dropout_p)
        self.frame8 = TimeDelayLayer_v5(input_dim=512, output_dim=512, context_size=1, dilation=1,
                                        activation=activation, batch_norm=batch_norm, dropout_p=dropout_p)
        self.frame9 = TimeDelayLayer_v5(input_dim=512, output_dim=512, context_size=1, dilation=1,
                                        activation=activation, batch_norm=batch_norm, dropout_p=dropout_p)
        self.frame10 = TimeDelayLayer_v5(input_dim=512, output_dim=1500, context_size=1, dilation=1,
                                         activation=activation, batch_norm=batch_norm, dropout_p=dropout_p)

        # self.segment11 = nn.Linear(3000, embedding_size)
        # self.leakyrelu = nn.LeakyReLU()
        # self.batchnorm = nn.BatchNorm1d(embedding_size)

        if encoder_type == 'STAP':
            self.encoder = StatisticPooling(input_dim=1500)
        else:
            self.encoder = nn.AdaptiveAvgPool2d((1, None))

        seg12 = [nn.Linear(3000, embedding_size)]
        seg13 = [nn.Linear(embedding_size, embedding_size)]
        if activation == 'relu':
            seg12.append(nn.ReLU())
            seg13.append(nn.ReLU())
        elif activation == 'leakyrelu':
            seg12.append(nn.LeakyReLU())
            seg13.append(nn.LeakyReLU())
        elif activation == 'prelu':
            seg12.append(nn.PReLU())
            seg13.append(nn.PReLU())
        seg12.append(nn.BatchNorm1d(embedding_size))
        seg13.append(nn.BatchNorm1d(embedding_size))

        self.segment12 = nn.Sequential(seg12)
        self.segment13 = nn.Sequential(seg13)

        self.classifier = nn.Linear(embedding_size, num_classes)

        for m in self.modules():  # 对于各层参数的初始化
            if isinstance(m, nn.BatchNorm1d):  # weight设置为1，bias为0
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def set_global_dropout(self, dropout_p):
        self.dropout_p = dropout_p

        for m in self.modules():
            if isinstance(m, ETDNN_v4):
                m.set_dropout(dropout_p)

    def forward(self, x):
        # pdb.set_trace()
        if len(x.shape) == 4:
            x = x.squeeze(1).float()

        if self.inst_layer != None:
            x = self.inst_layer(x)

        x = self.frame1(x)
        x = self.affine2(x)
        x = self.frame3(x)
        x = self.affine4(x)
        x = self.frame5(x)
        x = self.affine6(x)
        x = self.frame7(x)
        x = self.frame8(x)
        x = self.frame9(x)
        x = self.frame10(x)

        x = self.encoder(x)
        embeddings_a = self.segment12(x)
        embeddings_b = self.segment13(embeddings_a)

        logits = self.classifier(embeddings_b)

        return logits, embeddings_b

    def xvector(self, x):

        if self.inst_layer != None:
            x = self.inst_layer(x)

        x = self.frame1(x)
        x = self.affine2(x)
        x = self.frame3(x)
        x = self.affine4(x)
        x = self.frame5(x)
        x = self.affine6(x)
        x = self.frame7(x)
        x = self.frame8(x)
        x = self.frame9(x)
        x = self.frame10(x)

        x = self.encoder(x)
        embeddings_a = self.segment12(x)

        return embeddings_a
