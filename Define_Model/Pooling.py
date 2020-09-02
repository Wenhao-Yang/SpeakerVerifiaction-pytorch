#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: Pooling.py
@Time: 2020/4/15 10:57 PM
@Overview:
"""
import numpy as np
import torch
import torch.nn as nn


class SelfVadPooling(nn.Module):
    def __init__(self, input_dim):
        self.fc1 = nn.Linear(input_dim, 1, bias=False)
        self.activation = nn.Hardtanh(min_val=0., max_val=1.)
        nn.init.uniform(self.fc1.weight, a=0.8, b=1.2)

    def forward(self, x):
        # x_energy = torch.abs(self.fc1(x)).log()
        x_energy = self.fc1(x)
        x_weight = self.activation(x_energy)

        x_weight = 2. * x_weight - x_weight.pow(2)
        return x * x_weight


class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SelfAttentionPooling, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.attention_linear = nn.Linear(input_dim, self.hidden_dim)
        self.attention_activation = nn.Sigmoid()
        self.attention_vector = nn.Parameter(torch.rand(self.hidden_dim, 1))
        self.attention_soft = nn.Tanh()

    def forward(self, x):
        """
        :param x:   [batch, length, feat_dim] vector
        :return:   [batch, feat_dim] vector
        """
        x_shape = x.shape
        x = x.squeeze()
        if x_shape[0] == 1:
            x = x.unsqueeze(0)

        assert len(x.shape) == 3, print(x.shape)
        if x.shape[-2] == self.input_dim:
            x = x.transpose(-1, -2)
        # print(x.shape)
        fx = self.attention_activation(self.attention_linear(x))
        vf = fx.matmul(self.attention_vector)
        alpha = self.attention_soft(vf)

        alpha_ht = x.mul(alpha)
        mean = torch.sum(alpha_ht, dim=-2)

        return mean


class AttentionStatisticPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionStatisticPooling, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.attention_linear = nn.Linear(input_dim, self.hidden_dim)
        self.attention_activation = nn.Sigmoid()
        self.attention_vector = nn.Parameter(torch.rand(self.hidden_dim, 1))
        self.attention_soft = nn.Tanh()

    def forward(self, x):
        """
        :param x:   [length,feat_dim] vector
        :return:   [feat_dim] vector
        """
        x_shape = x.shape
        x = x.squeeze()
        if x_shape[0] == 1:
            x = x.unsqueeze(0)

        assert len(x.shape) == 3, print(x.shape)
        if x.shape[-2] == self.input_dim:
            x = x.transpose(-1, -2)

        fx = self.attention_activation(self.attention_linear(x))
        vf = fx.matmul(self.attention_vector)
        alpha = self.attention_soft(vf)

        alpha_ht = x.mul(alpha)
        mean = torch.sum(alpha_ht, dim=-2, keepdim=True)

        sigma_power = torch.sum(torch.pow(x - mean, 2).mul(alpha), dim=-2).add_(1e-12)
        # alpha_ht_ht = x*x.mul(alpha)
        sigma = torch.sqrt(sigma_power)

        mean_sigma = torch.cat((mean.squeeze(1), sigma), 1)

        return mean_sigma


class StatisticPooling(nn.Module):

    def __init__(self, input_dim):
        super(StatisticPooling, self).__init__()
        self.input_dim = input_dim

    def forward(self, x):
        """
        :param x:   [length,feat_dim] vector
        :return:   [feat_dim] vector
        """
        x_shape = x.shape
        x = x.squeeze()
        if x_shape[0] == 1:
            x = x.unsqueeze(0)

        assert len(x.shape) == 3, print(x.shape)
        if x.shape[-2] == self.input_dim:
            x = x.transpose(-1, -2)

        mean_x = x.mean(dim=1)
        std_x = x.var(dim=1, unbiased=False).add_(1e-12).sqrt()
        mean_std = torch.cat((mean_x, std_x), 1)
        return mean_std


class AdaptiveStdPool2d(nn.Module):
    def __init__(self, output_size):
        super(AdaptiveStdPool2d, self).__init__()
        self.output_size = output_size

    def forward(self, input):

        input_shape = input.shape
        if len(input_shape) == 3:
            input = input.unsqueeze(1)
            input_shape = input.shape

        assert len(input_shape) == 4, print(input.shape)
        output_shape = list(self.output_size)

        if output_shape[1] == None:
            output_shape[1] = input.shape[3]

        if output_shape[0] == None:
            output_shape[0] = input.shape[2]

        output_shape[0] = int(output_shape[0] / 2)
        # kernel_y = (input_shape[3] + self.output_size[1] - 1) // self.output_size[1]
        x_stride = input_shape[3] / output_shape[1]
        y_stride = input_shape[2] / output_shape[0]

        print(x_stride, y_stride)

        output = []

        for x_idx in range(output_shape[1]):
            x_output = []
            x_start = int(np.floor(x_idx * x_stride))

            x_end = int(np.ceil((x_idx + 1) * x_stride))
            for y_idx in range(output_shape[0]):
                y_start = int(np.floor(y_idx * y_stride))
                y_end = int(np.ceil((y_idx + 1) * y_stride))
                stds = input[:, :, y_start:y_end, x_start:x_end].var(dim=(2, 3), unbiased=False, keepdim=True).add_(
                    1e-14).sqrt()
                means = input[:, :, y_start:y_end, x_start:x_end].mean(dim=(2, 3), keepdim=True)
                # print(stds.shape)
                # stds = torch.std(input[:, :, y_start:y_end, x_start:x_end] , dim=2, )
                # sum_std = torch.sum(stds, dim=3, keepdim=True)

                x_output.append(means)
                x_output.append(stds)

            output.append(torch.cat(x_output, dim=2))
        output = torch.cat(output, dim=3)

        # print(output.isnan())

        return output
