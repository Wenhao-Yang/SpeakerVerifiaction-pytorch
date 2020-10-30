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
import torch.nn.functional as F


class SelfVadPooling(nn.Module):
    def __init__(self, input_dim, input_length=300):
        super(SelfVadPooling, self).__init__()
        self.conv1 = nn.Conv1d(1, 1, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(input_dim, 1)
        self.bn1 = nn.BatchNorm1d(1)
        #
        # self.conv2 = nn.Conv1d(1, 1, kernel_size=5, stride=1, padding=2)
        # self.fc2 = nn.Linear(input_length, 1)
        # self.bn2 = nn.BatchNorm1d(1)

        self.activation = nn.Hardtanh(min_val=0.001, max_val=1.0)
        # nn.init.constant(self.fc1.weight, 0.1)

    def forward(self, x):
        x_energy = self.fc1(x).squeeze(-1)  # .log()
        x_energy = self.bn1(x_energy)
        vad = self.conv1(x_energy).unsqueeze(-1)
        vad_weight = self.activation(vad)

        # x_freq = self.fc2(x.transpose(2, 3)).squeeze(-1).log()
        # x_freq = self.bn2(x_freq)
        # freq = self.conv2(x_freq).unsqueeze(2)
        # freq_weight = self.activation(freq)

        # x_weight = 2. * x_weight - x_weight.pow(2)
        return x * vad_weight  # * freq_weight


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
        mean = torch.sum(alpha_ht, dim=-2)

        # pdb.set_trace()
        sigma_power = torch.sum(torch.pow(x, 2).mul(alpha), dim=-2) - torch.pow(mean, 2)
        # alpha_ht_ht = x*x.mul(alpha)
        sigma = torch.sqrt(sigma_power.clamp(min=1e-12))

        mean_sigma = torch.cat((mean, sigma), 1)

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


# https://github.com/keetsky/Net_ghostVLAD-pytorch

class GhostVLAD_v1(nn.Module):
    def __init__(self, num_clusters=8, gost=1, dim=128, normalize_input=True):
        super(GhostVLAD_v1, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.gost = gost
        self.normalize_input = normalize_input
        self.fc = nn.Linear(dim, num_clusters + gost)
        self.centroids = nn.Parameter(torch.rand(num_clusters + gost, dim))
        self._init_params()

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight.data)
        nn.init.constant_(self.fc.bias.data, 0.0)

    def forward(self, x):
        '''
        x: N x D
        '''
        N, C = x.shape[:2]  # 10,128
        assert C == self.dim, "feature dim not correct"

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=0)

        soft_assign = self.fc(x).unsqueeze(0).permute(0, 2, 1)  # (N, C+g)->(1, N, C+g)->(1, C+g, N)
        soft_assign = F.softmax(soft_assign, dim=1)  # (1, C+g, N)

        # soft_assign=soft_assign[:,:self.num_clusters,:]#(1,8,10)
        x_flatten = x.unsqueeze(0).permute(0, 2, 1)  # x.view(1, C, N)

        residual = x_flatten.expand(self.num_clusters + self.gost, -1, -1, -1).permute(1, 0, 2, 3) - \
                   self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        # (1, c+g, C, N)

        residual *= soft_assign.unsqueeze(2)
        print(residual.shape)

        vlad = residual.sum(dim=-1)  # (1,9,128)
        vlad = vlad[:, :self.num_clusters, :]  # (1,8,128)
        vlad = F.normalize(vlad, p=2, dim=2)
        vlad = vlad.view(1, -1)
        vlad = F.normalize(vlad, p=2, dim=1)  # (1,8*128)

        return vlad


# https://github.com/taylorlu/Speaker-Diarization/blob/master/ghostvlad/model.py
class GhostVLAD_v2(nn.Module):
    def __init__(self, num_clusters=8, gost=1, dim=128, normalize_input=True):
        super(GhostVLAD_v2, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.gost = gost
        self.normalize_input = normalize_input
        self.fc = nn.Linear(dim, num_clusters + gost)
        self.centroids = nn.Parameter(torch.rand(num_clusters + gost, dim))
        self._init_params()

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight.data)
        nn.init.constant_(self.fc.bias.data, 0.0)

    def forward(self, x):
        '''
        x: N x D
        '''
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=-1)

        feat = x
        cluster_score = self.fc(x)  # bz x cluster

        # num_features = feat.shape[-1]
        # softmax normalization to get soft-assignment.
        # A : bz  x clusters
        max_cluster_score = torch.max(cluster_score, dim=-1, keepdim=True).values

        exp_cluster_score = torch.exp(cluster_score - max_cluster_score)
        A = exp_cluster_score / torch.sum(exp_cluster_score, dim=-1, keepdim=True)
        # Now, need to compute the residual, self.cluster: clusters x D
        A = A.unsqueeze(-1)  # A : bz x clusters x 1

        feat_broadcast = feat.unsqueeze(-2)  # feat_broadcast : bz x 1 x D
        feat_res = feat_broadcast - self.centroids  # feat_res : bz x clusters x D

        weighted_res = torch.mul(A, feat_res)  # weighted_res : bz x clusters x D
        # cluster_res = torch.sum(weighted_res, [1, 2])
        cluster_res = weighted_res
        cluster_res = cluster_res[:, :self.num_clusters, :]
        # cluster_l2 = torch.nn.functional.l2_normalize(cluster_res, -1)
        # outputs = cluster_res.reshape([-1, int(self.num_clusters) * int(num_features)])

        outputs = cluster_res.sum(dim=-2)

        return outputs


class LinearTransform(nn.Module):
    def __init__(self, dim=128, normalize_input=True):
        self.dim = dim
        self.normalize_input = normalize_input
        self.linear_trans = nn.Sequential(nn.Linear(dim, dim),
                                          nn.BatchNorm1d(dim),
                                          nn.ReLU())

    def forward(self, x):
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=-1)

        trans = self.linear_trans(x)

        return x + trans
