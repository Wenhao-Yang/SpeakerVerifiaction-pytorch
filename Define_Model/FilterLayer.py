#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: FilterLayer.py
@Time: 2020/8/19 20:30
@Overview:
"""

import numpy as np
import torch
from python_speech_features import hz2mel, mel2hz
from torch import nn


class fDLR(nn.Module):
    def __init__(self, input_dim, sr, num_filter):
        super(fDLR, self).__init__()
        self.input_dim = input_dim
        self.num_filter = num_filter
        self.sr = sr

        input_freq = np.linspace(0, self.sr / 2, input_dim)
        self.input_freq = nn.Parameter(torch.from_numpy(input_freq).expand(num_filter, input_dim).float(),
                                       requires_grad=False)

        centers = np.linspace(0, hz2mel(sr / 2), num_filter + 2)
        centers = mel2hz(centers)
        self.frequency_center = nn.Parameter(torch.from_numpy(centers[1:-1]).float().reshape(num_filter, 1))

        bandwidth = []
        for i in range(2, len(centers)):
            bandwidth.append(centers[i] - centers[i - 1])
        self.bandwidth = nn.Parameter(torch.tensor(bandwidth).reshape(num_filter, 1).float())
        self.gain = nn.Parameter(torch.ones(num_filter, dtype=torch.float32).reshape(num_filter, 1))

    def forward(self, input):
        frequency_center = self.frequency_center.sort(dim=0).values
        new_centers = frequency_center.expand(self.num_filter, self.input_dim)
        if input.is_cuda:
            new_centers = new_centers.cuda()

        # pdb.set_trace()
        power = -1. * torch.pow(self.input_freq - new_centers, 2)
        power = torch.div(power, 0.5 * self.bandwidth.pow(2))

        weights = torch.exp(power)
        weights = weights / weights.max(dim=1, keepdim=True).values

        weights = weights.mul(self.gain).transpose(0, 1)

        return torch.matmul(input, weights)
