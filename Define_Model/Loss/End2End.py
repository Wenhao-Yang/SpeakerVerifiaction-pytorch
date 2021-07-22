#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: End2End.py
@Time: 2021/7/22 11:57
@Overview:
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy

# https://github.com/clovaai/voxceleb_trainer
from TrainAndTest.common_func import correct_output


class AngleProtoLoss(nn.Module):

    def __init__(self, init_w=10.0, init_b=-5.0):
        super(AngleProtoLoss, self).__init__()
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.criterion = torch.nn.CrossEntropyLoss()

        print('Initialised AngleProto')

    def forward(self, x, label=None):
        out_anchor = torch.mean(x[:, 1:, :], 1)
        out_positive = x[:, 0, :]
        stepsize = out_anchor.size()[0]

        cos_sim_matrix = F.cosine_similarity(out_positive.unsqueeze(-1).expand(-1, -1, stepsize),
                                             out_anchor.unsqueeze(-1).expand(-1, -1, stepsize).transpose(0, 2))
        torch.clamp(self.w, 1e-6)
        cos_sim_matrix = cos_sim_matrix * self.w + self.b

        label = torch.from_numpy(numpy.asarray(range(0, stepsize))).cuda()
        loss = self.criterion(cos_sim_matrix, label)
        prec = correct_output(cos_sim_matrix.detach().cpu(), label.detach().cpu(), topk=(1,))

        return loss, prec[0]


class GE2ELoss(nn.Module):

    def __init__(self, init_w=10.0, init_b=-5.0):
        super(GE2ELoss, self).__init__()
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.criterion = torch.nn.CrossEntropyLoss()

        print('Initialised GE2E')

    def forward(self, x, label=None):
        gsize = x.size()[1]
        centroids = torch.mean(x, 1)
        stepsize = x.size()[0]

        cos_sim_matrix = []

        for ii in range(0, gsize):
            idx = [*range(0, gsize)]
            idx.remove(ii)
            exc_centroids = torch.mean(x[:, idx, :], 1)
            cos_sim_diag = F.cosine_similarity(x[:, ii, :], exc_centroids)
            cos_sim = F.cosine_similarity(x[:, ii, :].unsqueeze(-1).expand(-1, -1, stepsize),
                                          centroids.unsqueeze(-1).expand(-1, -1, stepsize).transpose(0, 2))
            cos_sim[range(0, stepsize), range(0, stepsize)] = cos_sim_diag
            cos_sim_matrix.append(torch.clamp(cos_sim, 1e-6))

        cos_sim_matrix = torch.stack(cos_sim_matrix, dim=1)

        torch.clamp(self.w, 1e-6)
        cos_sim_matrix = cos_sim_matrix * self.w + self.b

        label = torch.from_numpy(numpy.asarray(range(0, stepsize))).cuda()
        nloss = self.criterion(cos_sim_matrix.view(-1, stepsize),
                               torch.repeat_interleave(label, repeats=gsize, dim=0).cuda())
        prec = correct_output(cos_sim_matrix.view(-1, stepsize).detach().cpu(),
                              torch.repeat_interleave(label, repeats=gsize, dim=0).detach().cpu(), topk=(1,))

        return nloss, prec[0]


class ProtoLoss(nn.Module):

    def __init__(self):
        super(ProtoLoss, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss()

        print('Initialised Prototypical Loss')

    def forward(self, x, label=None):
        out_anchor = torch.mean(x[:, 1:, :], 1)
        out_positive = x[:, 0, :]
        stepsize = out_anchor.size()[0]

        output = -1 * (F.pairwise_distance(out_positive.unsqueeze(-1).expand(-1, -1, stepsize),
                                           out_anchor.unsqueeze(-1).expand(-1, -1, stepsize).transpose(0, 2)) ** 2)
        label = torch.from_numpy(numpy.asarray(range(0, stepsize))).cuda()
        nloss = self.criterion(output, label)
        prec = correct_output(output.detach().cpu(), label.detach().cpu(), topk=(1,))

        return nloss, prec[0]
