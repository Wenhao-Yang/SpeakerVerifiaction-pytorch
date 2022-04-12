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
import random
from sklearn import metrics

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


def tuneThresholdfromScore(scores, labels, target_fa, target_fr=None):
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    fnr = fnr * 100
    fpr = fpr * 100

    tunedThreshold = [];
    if target_fr:
        for tfr in target_fr:
            idx = numpy.nanargmin(numpy.absolute((tfr - fnr)))
            tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]]);

    for tfa in target_fa:
        idx = numpy.nanargmin(numpy.absolute((tfa - fpr)))  # numpy.where(fpr<=tfa)[0][-1]
        tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]]);

    idxE = numpy.nanargmin(numpy.absolute((fnr - fpr)))
    eer = max(fpr[idxE], fnr[idxE])

    return (tunedThreshold, eer, fpr, fnr);


class PairwiseLoss(nn.Module):

    def __init__(self, loss_func=None, hard_rank=0, hard_prob=0, margin=0):
        super(PairwiseLoss, self).__init__()
        self.loss_func = loss_func
        self.hard_rank = hard_rank
        self.hard_prob = hard_prob
        self.margin = margin

        print('Initialised Pairwise Loss')

    def forward(self, x, label=None):

        out_anchor = x[:, 0, :]
        out_positive = x[:, 1, :]
        stepsize = out_anchor.size()[0]

        output = -1 * (F.pairwise_distance(out_anchor.unsqueeze(-1).expand(-1, -1, stepsize),
                                           out_positive.unsqueeze(-1).expand(-1, -1, stepsize).transpose(0, 2)) ** 2)

        negidx = self.mineHardNegative(output.detach())

        out_negative = out_positive[negidx, :]

        labelnp = numpy.array([1] * len(out_positive) + [0] * len(out_negative))

        ## calculate distances
        pos_dist = F.pairwise_distance(out_anchor, out_positive)
        neg_dist = F.pairwise_distance(out_anchor, out_negative)

        ## loss functions
        if self.loss_func == 'Contrastive':
            nloss = torch.mean(torch.cat([torch.pow(pos_dist, 2), torch.pow(F.relu(self.margin - neg_dist), 2)], dim=0))
        elif self.loss_func == 'Triplet':
            nloss = torch.mean(F.relu(torch.pow(pos_dist, 2) - torch.pow(neg_dist, 2) + self.margin))

        scores = -1 * torch.cat([pos_dist, neg_dist], dim=0).detach().cpu().numpy()

        errors = tuneThresholdfromScore(scores, labelnp, []);

        return nloss, errors[1]

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Hard negative mining
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def mineHardNegative(self, output):

        negidx = []

        for idx, similarity in enumerate(output):

            simval, simidx = torch.sort(similarity, descending=True)

            if self.hard_rank < 0:

                ## Semi hard negative mining

                semihardidx = simidx[(similarity[idx] - self.margin < simval) & (simval < similarity[idx])]

                if len(semihardidx) == 0:
                    negidx.append(random.choice(simidx))
                else:
                    negidx.append(random.choice(semihardidx))

            else:

                ## Rank based negative mining

                simidx = simidx[simidx != idx]

                if random.random() < self.hard_prob:
                    negidx.append(simidx[random.randint(0, self.hard_rank)])
                else:
                    negidx.append(random.choice(simidx))

        return negidx
