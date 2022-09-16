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
import torch.nn.functional as F


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


class DistributeLoss(nn.Module):
    """Distribute of Distance loss.

    """

    def __init__(self, stat_type="mean", margin=0.2, p_target=0.1):
        super(DistributeLoss, self).__init__()
        self.stat_type = stat_type
        self.margin = margin
        self.p_target = p_target

    def forward(self, dist, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        # norms = self.centers.data.norm(p=2, dim=1, keepdim=True).add(1e-14)
        # self.centers.data = self.centers.data / norms * self.alpha

        if len(labels.shape) == 1:
            labels = labels.unsqueeze(1)

        positive_dist = dist.gather(dim=1, index=labels)

        negative_label = torch.arange(dist.shape[1]).reshape(1, -1).repeat(positive_dist.shape[0], 1)
        if labels.is_cuda:
            negative_label = negative_label.cuda()

        negative_label = negative_label.scatter(1, labels, -1)
        negative_label = torch.where(negative_label != -1)[1].reshape(positive_dist.shape[0], -1)

        negative_dist = dist.gather(dim=1, index=negative_label)

        mean = positive_dist.mean()  # .clamp_min(0)

        if self.stat_type == "stddmean":
            loss = positive_dist.std() / mean.clamp(min=1e-6)
            loss = loss ** 2

        elif self.stat_type == "kurtoses":
            diffs = positive_dist - mean
            var = torch.mean(torch.pow(diffs, 2.0))
            std = torch.pow(var, 0.5)
            z_scores = diffs / std

            kurtoses = torch.mean(torch.pow(z_scores, 4.0)) - 3.0
            # skewness = torch.mean(torch.pow(z_scores, 3.0))
            loss = (-kurtoses).clamp_min(0)
        elif self.stat_type == "margin":
            positive_theta = torch.acos(positive_dist)
            loss = (2 * positive_theta - self.margin).clamp_min(0).mean()
        elif self.stat_type == "margin1":
            positive_theta = torch.acos(positive_dist)
            loss = (positive_theta - self.margin).clamp_min(0).mean()
        elif self.stat_type == "margin1sum":
            positive_theta = torch.acos(positive_dist)
            loss = (positive_theta - self.margin).clamp_min(0).sum()
        elif self.stat_type == "marginsum":
            positive_theta = torch.acos(positive_dist)
            loss = (2 * positive_theta - self.margin).clamp_min(0).sum()
        elif self.stat_type == "maxmargin":
            positive_theta = torch.acos(positive_dist)
            # loss = (2 * positive_theta - self.margin).clamp_min(0).max()
            loss = (positive_theta - self.margin).clamp_min(0).max()
        elif self.stat_type == "maxnegative":
            negative_theta = torch.acos(negative_dist)
            # loss = (2 * positive_theta - self.margin).clamp_min(0).max()
            loss = (0.5 * np.pi - negative_theta - self.margin).clamp_min(0).max()

        elif self.stat_type == "mindcf":
            positive_theta = torch.acos(positive_dist)
            negative_theta = torch.acos(negative_dist)

            loss = self.p_target * positive_theta.clamp_min(self.margin).max() + (
                    self.p_target - 1) * negative_theta.clamp_max(0.5 * np.pi - self.margin).min()

        return loss

    def __repr__(self):
        return "DistributeLoss(margin=%f, stat_type=%s, self.p_target=%s)" % (
            self.margin, self.stat_type, self.p_target)


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
        batch_size_s = int(source.size()[0])
        batch_size_t = int(target.size()[0])

        kernels = guassian_kernel(source, target,
                                  kernel_mul=self.kernel_mul,
                                  kernel_num=self.kernel_num,
                                  fix_sigma=self.fix_sigma)

        XX = kernels[:batch_size_s, :batch_size_s].mean()
        YY = kernels[batch_size_t:, batch_size_t:].mean()
        XY = kernels[:batch_size_s, batch_size_t:].mean()
        YX = kernels[batch_size_t:, :batch_size_s].mean()

        loss = torch.sum(XX + YY - XY - YX)
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


class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes = 3, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """

        super(focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha)==num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            print("Focal_loss alpha = {}, 将对每一类权重进行精细化赋值".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1   #如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            print(" --- Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用 --- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1,preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_softmax = torch.nn.functional.softmax(preds, dim=1) # 这里并没有直接使用log_softmax, 因为后面会用到softmax的结果(当然你也可以使用log_softmax,然后进行exp操作)
        preds_logsoft = torch.log(preds_softmax)
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))  # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        self.alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class pAUCLoss(nn.Module):

    def __init__(self, s=10.0, margin=0.2):
        super(pAUCLoss, self).__init__()
        self.margin = margin
        self.s = s

    def forward(self, target, nontarget):
        loss = self.margin - (target.repeat(nontarget.shape[0]) - nontarget.repeat(target.shape[0]))
        loss = loss.clamp_min(0)  # .reshape(target.shape[0], nontarget.shape[0]) * self.s
        # print(loss.shape)
        # loss = loss.max(dim=1)[0]

        loss = torch.mean(loss.pow(2))

        return loss


class aAUCLoss(nn.Module):

    def __init__(self, s=10.0, margin=0.2):
        super(pAUCLoss, self).__init__()
        self.margin = margin
        self.s = s

    def forward(self, costh, label):
        label = label.reshape(-1, 1)
        positive_dist = costh.gather(dim=1, index=label)

        negative_label = torch.arange(costh.shape[1]).reshape(1, -1).repeat(positive_dist.shape[0], 1)
        if label.is_cuda:
            negative_label = negative_label.cuda()

        negative_label = negative_label.scatter(1, label, -1)
        negative_label = torch.where(negative_label != -1)[1].reshape(positive_dist.shape[0], -1)
        negative_dist = costh.gather(dim=1, index=negative_label)

        loss = torch.sigmoid(self.s * (positive_dist - negative_dist)).mean()

        return loss


class aDCFLoss(nn.Module):
    def __init__(self, alpha=40, beta=0.25, gamma=0.75, omega=0.5):
        super(aDCFLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.omega = nn.Parameter(torch.tensor(omega))
        # self.ce = nn.CrossEntropyLoss()
        self.gamma = gamma

    def forward(self, costh, label):
        label = label.reshape(-1, 1)
        positive_dist = costh.gather(dim=1, index=label)

        negative_label = torch.arange(costh.shape[1]).reshape(1, -1).repeat(positive_dist.shape[0], 1)
        if label.is_cuda:
            negative_label = negative_label.cuda()

        negative_label = negative_label.scatter(1, label, -1)
        negative_label = torch.where(negative_label != -1)[1].reshape(positive_dist.shape[0], -1)
        negative_dist = costh.gather(dim=1, index=negative_label)

        pfa = self.gamma * torch.sigmoid(self.alpha * (positive_dist - self.omega)).mean()
        pmiss = self.beta * torch.sigmoid(self.alpha * (self.omega - negative_dist)).mean()

        loss = pfa + pmiss

        return loss


class AttentionTransferLoss(nn.Module):
    def __init__(self, attention_type='both', norm_type='input'):
        super(AttentionTransferLoss, self).__init__()
        self.attention_type = attention_type
        self.norm_type = norm_type

    def at(self, x):
        if self.attention_type == 'both':
            return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))
        elif self.attention_type == 'time':
            return F.normalize(x.pow(2).mean(1).mean(2).view(x.size(0), -1))
        elif self.attention_type == 'freq':
            return F.normalize(x.pow(2).mean(1).mean(1).view(x.size(0), -1))

    def min_max(self, x):
        return (x - x.min()) / (x.max() - x.min())

    def forward(self, s_feats, t_feats):
        loss = 0.
        if self.norm_type == 'input':
            for s_f, t_f in zip(s_feats, t_feats):
                loss += (self.at(s_f) - self.at(t_f)).pow(2).mean()

        else:
            ups = nn.UpsamplingBilinear2d(s_feats[0].shape[-2:])
            s_map = torch.zeros_like(s_feats[0].mean(dim=1, keepdim=True))
            t_map = torch.zeros_like(t_feats[0].mean(dim=1, keepdim=True))

            for i, (s_f, t_f) in enumerate(zip(s_feats, t_feats)):
                weight = ((1 + i) / len(s_feats)) if 'weight' in self.norm_type else 1.0

                s_input = ups(s_f).mean(dim=1, keepdim=True).clamp_min(0)
                # s_input /= s_input.max()
                s_max = s_input.view(s_input.size(0), -1).max(dim=1).values.reshape(-1, 1, 1, 1)
                s_map += weight * s_input / s_max

                t_input = ups(t_f).mean(dim=1, keepdim=True).clamp_min(0)
                t_max = t_input.view(t_input.size(0), -1).max(dim=1).values.reshape(-1, 1, 1, 1)
                # t_input /= t_input.max()
                t_map += weight * t_input / t_max

            t_map = t_map / len(t_feats)
            s_map = s_map / len(s_feats)

            if self.attention_type == 'both':
                # loss += (self.min_max(s_map.mean(dim=2, keepdim=True)) - self.min_max(
                #     t_map.mean(dim=2, keepdim=True))).pow(2).mean()
                # loss += (self.min_max(s_map.mean(dim=3, keepdim=True)) - self.min_max(
                #     t_map.mean(dim=3, keepdim=True))).pow(2).mean()

                loss += (s_map.mean(dim=2, keepdim=True) - t_map.mean(dim=2, keepdim=True)).pow(2).mean()
                loss += (s_map.mean(dim=3, keepdim=True) - t_map.mean(dim=3, keepdim=True)).pow(2).mean()

                loss = loss / 2

            elif self.attention_type == 'time':
                # loss += (self.min_max(s_map.mean(dim=3, keepdim=True)) - self.min_max(
                #     t_map.mean(dim=3, keepdim=True))).pow(2).mean()
                loss += (s_map.mean(dim=3, keepdim=True) - t_map.mean(dim=3, keepdim=True)).pow(2).mean()

            elif self.attention_type == 'freq':
                # loss += (self.min_max(s_map.mean(dim=2, keepdim=True)) - self.min_max(
                #     t_map.mean(dim=2, keepdim=True))).pow(2).mean()
                loss += (s_map.mean(dim=2, keepdim=True) - t_map.mean(dim=2, keepdim=True)).pow(2).mean()

        return loss
