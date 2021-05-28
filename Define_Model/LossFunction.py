#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: LossFunction.py
@Time: 2020/1/8 3:46 PM
@Overview:
"""

import torch
import torch.nn as nn
from geomloss import SamplesLoss
import numpy as np

class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=2):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim

        centers = torch.randn(self.num_classes, self.feat_dim)
        centers = torch.nn.functional.normalize(centers, p=2, dim=1)
        alpha = np.ceil(np.log(0.99 * (num_classes - 2) / (1 - 0.99)))
        centers *= np.sqrt(alpha)

        self.centers = nn.Parameter(centers)

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        # norms = self.centers.data.norm(p=2, dim=1, keepdim=True).add(1e-14)
        # self.centers.data = self.centers.data / norms * self.alpha

        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        # if self.use_gpu: classes = classes.cuda()
        if self.centers.is_cuda:
            classes = classes.cuda()

        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.mean()

        return loss


class VarianceLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=2):
        super(VarianceLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim

        centers = torch.randn(self.num_classes, self.feat_dim)
        centers = torch.nn.functional.normalize(centers, p=2, dim=1)
        alpha = np.ceil(np.log(0.99 * (num_classes - 2) / (1 - 0.99)))
        centers *= np.sqrt(alpha)

        self.centers = nn.Parameter(centers)

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        # norms = self.centers.data.norm(p=2, dim=1, keepdim=True).add(1e-14)
        # self.centers.data = self.centers.data / norms * self.alpha

        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        # if self.use_gpu: classes = classes.cuda()
        if self.centers.is_cuda:
            classes = classes.cuda()

        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.mean() + dist.std()

        return loss


class MultiCenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True, alpha=10., num_center=1):
        super(MultiCenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        self.alpha = alpha
        self.num_center = num_center

        if self.num_center > 1:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.num_center, self.feat_dim).cuda())  #
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())  # .cuda()

        # self.centers.data.uniform_(-1, 1).renorm_(2, 1, 1e-8).mul_(1e8)

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        # norms = self.centers.data.norm(p=2, dim=1, keepdim=True).add(1e-14)
        if self.alpha:
            norms = self.centers.data.pow(2).sum(dim=1, keepdim=True).add(1e-12).sqrt()
            self.centers.data = torch.div(self.centers.data, norms) * self.alpha

        batch_size = x.size(0)

        if self.num_center == 1:
            distmat = torch.pow(x, 2).sum(dim=-1, keepdim=True).expand(batch_size, self.num_classes) + \
                      torch.pow(self.centers, 2).sum(dim=-1, keepdim=True).expand(self.num_classes, batch_size).t()
            distmat.addmm_(1, -2, x, self.centers.t())
        else:

            distmat = torch.pow(x, 2).sum(dim=-1, keepdim=True).expand(batch_size, self.num_classes).unsqueeze(
                1).expand(batch_size, self.num_center, self.num_classes) + \
                      torch.pow(self.centers, 2).sum(dim=-1, keepdim=True).expand(self.num_classes, self.num_center,
                                                                                  batch_size).transpose(0, -1)

            distmat -= 2 * self.centers.matmul(x.t()).transpose(0, -1)
            distmat = distmat.min(dim=1).values

        classes = torch.arange(self.num_classes).long()

        if x.is_cuda:
            classes = classes.cuda()

        expand_labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = expand_labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()

        # pdb.set_trace()
        dist = dist.sum(dim=1)  # .add(1e-14).sqrt()
        loss = dist.clamp(min=1e-12, max=1e+12).mean()  # / int(self.partion*batch_size)

        return loss


class CenterCosLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True, alpha=0.0):
        super(CenterCosLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        self.alpha = alpha

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

        self.centers.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        if self.alpha:
            norms = self.centers.data.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12, max=1e+12)
            self.centers.data = self.centers.data / norms * self.alpha

        batch_size = x.size(0)
        # pdb.set_trace()
        classes = self.centers.index_select(dim=0, index=labels)
        all_cos = torch.cosine_similarity(x, classes)

        # lab_class = len(torch.unique(labels))
        dist = torch.exp(-3.5 * (all_cos - 1))
        loss = dist.sum() / batch_size

        return loss


class TupleLoss(nn.Module):

    def __init__(self, batch_size, tuple_size):
        super(TupleLoss, self).__init__()
        self.batch_size = batch_size
        self.tuple_size = tuple_size
        self.sim = nn.CosineSimilarity(dim=0, eps=1e-6)

    def forward(self, spk_representation, labels):
        """
        Args:
            x: (bashsize*tuplesize, dimension of linear layer)
            labels: ground truth labels with shape (batch_size).
        """
        feature_size = spk_representation.shape[1]
        w = torch.reshape(spk_representation, [self.batch_size, self.tuple_size, feature_size])

        loss = 0
        for indice_bash in range(self.batch_size):
            wi_enroll = w[indice_bash, 1:]  # shape:  (tuple_size-1, feature_size)
            wi_eval = w[indice_bash, 0]
            c_k = torch.mean(wi_enroll, dim=0)  # shape: (feature_size)
            # norm_c_k = c_k / torch.norm(c_k, p=2, keepdim=True)
            # normlize_ck = torch.norm(c_k, p=2, dim=0)
            # normlize_wi_eval = torch.norm(wi_eval, p=2, dim=0)
            cos_similarity = self.sim(c_k, wi_eval)
            score = cos_similarity

            loss += torch.sigmoid(score) * labels[indice_bash] + \
                    (1 - torch.sigmoid(score) * (1 - labels[indice_bash]))

        return -torch.log(loss / self.batch_size)


class RingLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, ring=10):
        super(RingLoss, self).__init__()
        self.ring = nn.Parameter(torch.tensor([float(ring)]))

    def forward(self, x):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        norms = torch.norm(x, p=2, dim=1)
        ring_loss = (norms - self.ring).pow(2).mean()

        return ring_loss


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    将源域数据和目标域数据转化为核矩阵，即上文中的K
    Params:
	    source: 源域数据（n * len(x))
	    target: 目标域数据（m * len(y))
	    kernel_mul:
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 不同高斯核的sigma值
	Return:
		sum(kernel_val): 多个核矩阵之和
    '''
    n_samples = int(source.size()[0]) + int(target.size()[0])  # 求矩阵的行数，一般source和target的尺度是一样的，这样便于计算
    total = torch.cat([source, target], dim=0)  # 将source,target按列方向合并
    # 将total复制（n+m）份
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    # 将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    # 求任意两个数据之间的和，得到的矩阵中坐标（i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0）
    L2_distance = ((total0 - total1) ** 2).sum(2)
    # 调整高斯核函数的sigma值
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    # 以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    # 高斯核函数的数学表达式
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    # 得到最终的核矩阵
    return sum(kernel_val)  # /len(kernel_val)


class MMD_Loss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super(MMD_Loss, self).__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = guassian_kernel(source, target,
                                  kernel_mul=self.kernel_mul,
                                  kernel_num=self.kernel_num,
                                  fix_sigma=self.fix_sigma)

        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return loss


class Wasserstein_Loss(nn.Module):
    def __init__(self, source_cls=1951):
        super(Wasserstein_Loss, self).__init__()
        self.source_cls = source_cls
        self.loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)

    def forward(self, feats, label):
        # pdb.set_trace()
        idx = torch.nonzero(torch.lt(label, self.source_cls)).squeeze()
        if len(idx) == 0:
            return self.loss(feats, feats)

        vectors_s = feats.index_select(dim=0, index=idx)

        idx = torch.nonzero(torch.ge(label, self.source_cls)).squeeze()
        if len(idx) == 0:
            return self.loss(feats, feats)

        vectors_t = feats.index_select(dim=0, index=idx)

        return self.loss(vectors_s, vectors_t)


class AttentionMining(nn.Module):
    def __init__(self):
        super(AttentionMining, self).__init__()

    def forward(self, x, label):
        x_shape = x.shape
        x = torch.nn.functional.log_softmax(x, dim=1)
        label = label.reshape(x_shape[0], 1)
        score_c = x.gather(1, label)
        score_c = torch.nn.functional.sigmoid(score_c)

        return score_c.mean()
