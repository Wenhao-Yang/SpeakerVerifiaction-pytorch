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
import torch
import torch.nn as nn


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
