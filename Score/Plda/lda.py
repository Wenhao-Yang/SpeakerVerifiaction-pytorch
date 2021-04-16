#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: lda.py
@Time: 2020/12/30 09:48
@Overview:
"""
import numpy as np

from Score.Plda.plda import ApplyFloor


class CovarianceStats(object):
    def __init__(self, dim):
        self.tot_covar_ = np.zeros((dim, dim))  # 总方差
        self.between_covar_ = np.zeros((dim, dim))  # 类间方差
        self.num_spk_ = 0
        self.num_utt_ = 0

    # get total covariance, normalized per number of frames.
    # 返回总方差
    def GetTotalCovar(self):
        assert (self.num_utt_ > 0)
        tot_covar = 1 / self.num_utt_ * self.tot_covar_
        return tot_covar

    # 返回类内方差
    def GetWithinCovar(self):
        assert (self.num_utt_ - self.num_spk_ > 0);
        within_covar = self.tot_covar_ - self.between_covar_
        within_covar *= 1.0 / self.num_utt_

        return within_covar

    # 更新统计量包括：total_var, spk_average, between_var
    def AccStats(self, utts_of_this_spk):
        num_utts = len(utts_of_this_spk)

        self.tot_covar_ += np.matmul(utts_of_this_spk.transpose(), utts_of_this_spk)
        spk_average = utts_of_this_spk.mean(axis=0)

        self.between_covar_ += num_utts * np.matmul(spk_average.reshape(-1, 1), spk_average.reshape(1, -1))
        self.num_utt_ += num_utts
        self.num_spk_ += 1

    # Will return Empty() if the within-class covariance matrix would be zero.
    def SingularTotCovar(self):
        return (self.num_utt_ < self.Dim())

    def Empty(self):
        return ((self.num_utt_ - self.num_spk_) == 0)

    def Info(self):
        info_str = "%d speakers, %d utterances. " % (self.num_spk_, self.num_utt_)
        return info_str

    def Dim(self):
        return len(self.tot_covar_)

    # Use default constructor and assignment operator.
    def AddStats(self, other):
        self.tot_covar_ += other.tot_covar_
        self.between_covar_ += other.between_covar_
        self.num_spk_ += other.num_spk_
        self.num_utt_ += other.num_utt_


def ComputeAndSubtractMean(utt2ivector):
    keys = list(utt2ivector.keys())
    dim = utt2ivector[keys[0]].shape

    num_ivectors = len(utt2ivector)
    mean = np.zeros(dim)

    for utt in utt2ivector:
        mean += 1.0 / num_ivectors * utt2ivector[utt]

    for utt in utt2ivector:
        utt2ivector[utt] -= mean

    return mean


def SubtractGlobalMean(utt2ivector):
    keys = list(utt2ivector.keys())
    dim = utt2ivector[keys[0]].shape

    num_ivectors = len(utt2ivector)
    mean = np.zeros(dim)

    for utt in utt2ivector:
        mean += 1.0 / num_ivectors * utt2ivector[utt]  # .astype(np.float32)

    print("Norm of iVector mean was ", np.sqrt(np.square(mean).sum()))
    for utt in utt2ivector:
        utt2ivector[utt] -= mean

    return mean


# LDA变换
def ComputeLdaTransform(utt2ivector, spk2utt, total_covariance_factor,
                        covariance_floor, lda_out):
    assert (len(utt2ivector) > 0);
    # lda后的维度
    lda_dim = lda_out.shape[0]
    dim = lda_out.shape[1]

    first_utt = list(utt2ivector.keys())[0]
    assert (dim == utt2ivector[first_utt].shape[-1])

    assert (lda_dim > 0 & lda_dim <= dim);
    stats = CovarianceStats(dim)

    # 使用输入的ivectors计算lda的统计量到stats
    for spk in spk2utt:
        uttlist = spk2utt[spk]
        N = len(uttlist)  # number of utterances.
        assert N > 0

        utts_of_this_spk = [utt2ivector[utt] for utt in uttlist]
        stats.AccStats(np.array(utts_of_this_spk))

    print("Stats have ", stats.Info())
    assert (not stats.Empty())
    assert (not stats.SingularTotCovar()), print("Too little data for iVector dimension.")

    # 总方差矩阵
    total_covar = stats.GetTotalCovar()

    # 类内方差
    within_covar = stats.GetWithinCovar()

    # mat_to_norm = factor*total_cov + (1-factor)*within_cov
    mat_to_normalize = total_covariance_factor * total_covar
    mat_to_normalize += (1.0 - total_covariance_factor) * within_covar

    # 分解mat_to_norm矩阵得到T
    T = ComputeNormalizingTransform(mat_to_normalize, covariance_floor)
    # print("T: ", str(T))

    # between_cov = total_cov - within_cov
    between_covar = total_covar - within_covar
    # print("between_covar: ", str(between_covar)) --

    #  between_cov_pro = between_covar
    between_covar_proj = np.matmul(T, between_covar).__matmul__(T.transpose())
    # print("between_covar_proj: ", str(between_covar_proj))

    # 分解between_cov_pro为特征向量s, 和矩阵U
    s, U = np.linalg.eig(between_covar_proj)
    # print("U before sort: ", str(U))

    sort_on_absolute_value = False  # any negative ones will go last (they
    # shouldn't exist anyway so doesn't
    # really matter)
    s_idx = np.flipud(np.argsort(s))

    for x in s:
        if x < 0:
            sort_on_absolute_value = False
            break

    s = s[s_idx]
    U = U.transpose()[s_idx].transpose()

    print("Singular values of between-class covariance after projecting " \
          "with interpolated [total/within] covariance with a weight of ",
          total_covariance_factor,
          " on the total covariance, are: \n", s)

    # U^T is the transform that will diagonalize the between-class covariance.
    #  U_part is just the part of U that corresponds to the kept dimensions.

    U_part = U[0:dim, 0:lda_dim]
    # We first transform by T and then by U_part^T.  This means T goes on the right.
    #  lda_out = U_part^T * temp * T (?)
    return np.matmul(U_part.T, T)


def ComputeNormalizingTransform(covar, floor):
    # 计算方差的特征值
    s, U = np.linalg.eig(covar)
    # print("Before sorted U is ", U)
    # 特征值排序
    # Sort eigvenvalues from largest to smallest.
    s_idx = np.flipud(np.argsort(s))
    s = s[s_idx]
    U = U.transpose()[s_idx].transpose()  # U[:,i] is the eigenvector corresponding to the eigenvalue s[i]

    # 取值大于某一floor的特征值
    # Floor eigenvalues to a small positive value.
    floor *= s[0]  # Floor relative to the largest eigenvalue
    floor = np.double(floor)
    num_floored = ApplyFloor(s, floor)

    if (num_floored > 0):
        print("Floored ", num_floored, " eigenvalues of covariance to ", floor)

    # Next two lines computes projection proj, such that
    # proj * covar * proj^T = I.
    # 计算投影矩阵
    s = np.power(s, -0.5)
    proj = np.matmul(np.identity(len(s)) * s, U.transpose())

    return proj
