#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: plda.py
@Time: 2019/12/6 上午10:42
@Overview: Modified from kaldi. The original script is 'kaldi/src/ivector/plda.h kaldi/src/ivector/plda.cc'
"""
import os
import struct

import numpy as np
from kaldi_io.kaldi_io import _read_vec_flt_binary, _read_mat_binary
from tqdm import tqdm

M_LOG_2PI = 1.8378770664093454835606594728112


def ApplyFloor(array_a, floor_val):
    n = 0

    for i in range(len(array_a)):
        if array_a[i] < floor_val:
            array_a[i] = floor_val
            n += 1
    return n


def Resize(shape):
    return np.zeros(shape=shape)


def write_vec_binary(fd, v):
    if v.dtype == 'float32':
        fd.write('FV '.encode())
    elif v.dtype == 'float64':
        fd.write('DV '.encode())
    else:
        raise Exception("'%s', please use 'float32' or 'float64'" % v.dtype)
    # Dim,
    fd.write('\04'.encode())
    fd.write(struct.pack(np.dtype('uint32').char, v.shape[0]))  # dim
    # Data,
    fd.write(v.tobytes())


def write_mat_binary(fd, m):
    if m.dtype == 'float32':
        fd.write('FM '.encode())
    elif m.dtype == 'float64':
        fd.write('DM '.encode())
    else:
        raise Exception("'%s', please use 'float32' or 'float64'" % m.dtype)
    # Dims,
    fd.write('\04'.encode())
    fd.write(struct.pack(np.dtype('uint32').char, m.shape[0]))  # rows
    fd.write('\04'.encode())
    fd.write(struct.pack(np.dtype('uint32').char, m.shape[1]))  # cols
    # Data,
    fd.write(m.tobytes())


class PldaConfig(object):
    """
    normalize_length: "If true, do length normalization as part of PLDA (see "
                   "code for details).  This does not set the length unit; "
                   "by default it instead ensures that the inner product "
                   "with the PLDA model's inverse variance (which is a "
                   "function of how many utterances the iVector was averaged "
                   "over) has the expected value, equal to the iVector "
                   "dimension."
    simple_length_norm: "If true, replace the default length normalization by an "
                   "alternative that normalizes the length of the iVectors to "
                   "be equal to the square root of the iVector dimension.
    """

    def __init__(self):
        self.normalize_length = True
        self.simple_length_norm = False

    def register(self, **kwargs):
        if 'normalize_length' in kwargs.keys():
            self.normalize_length = kwargs['normalize_length']
        if 'simple_length_norm' in kwargs.keys():
            self.simple_length_norm = kwargs['simple_length_norm']


class PLDA(object):

    def __init__(self):

        self.mean_ = None
        self.transform_ = None
        self.psi_ = None
        self.offset_ = None

    def Dim(self):
        return len(self.mean_)

    def ComputeDerivedVars(self):

        assert (self.Dim() > 0)
        # self.offset_.re(Dim())
        self.offset_ = -1.0 * np.matmul(self.transform_, self.mean_.reshape(-1, 1))

    def GetNormalizationFactor(self, transformed_ivector, num_examples):
        assert (num_examples > 0)
        #  Work out the normalization factor. The covariance for an average over "num_examples"
        #  training iVectors equals \Psi + I / num_examples.
        transformed_ivector_sq = np.power(transformed_ivector, 2)
        # print(transformed_ivector)
        #  inv_covar will equal 1.0 / (\Psi + I / num_examples).
        inv_covar = self.psi_ + 1.0 / num_examples
        # print(self.psi_.shape)
        inv_covar = 1.0 / inv_covar
        # print(np.linalg.norm(transformed_ivector_sq))
        # "transformed_ivector" should have covariance(\Psi + I / num_examples), i.e.
        # within-class /num_examples plus between- class covariance.So
        # transformed_ivector_sq.(I / num_examples + \Psi) ^ {-1} should be equal to
        # the dimension.
        dot_prod = np.matmul(inv_covar.reshape(1, -1), transformed_ivector_sq)
        # print(self.Dim(), dot_prod)
        return np.sqrt(self.Dim() / dot_prod)

    def TransformIvector(self, config, ivector, num_examples):

        assert (len(ivector) == self.Dim())
        transformed_ivector = self.offset_.copy()
        # print(self.mean_)
        # print(self.transform_)
        # print(self.offset_)
        transformed_ivector += np.matmul(self.transform_, ivector.reshape(-1, 1))  # matmul
        transformed_ivector
        if (config.simple_length_norm):
            normalization_factor = np.sqrt(transformed_ivector.shape[0]) / np.sqrt(np.square(transformed_ivector).sum())
        else:
            normalization_factor = self.GetNormalizationFactor(transformed_ivector, num_examples);

        if (config.normalize_length):
            transformed_ivector *= normalization_factor
        # print(normalization_factor)
        return transformed_ivector, normalization_factor

    def LogLikelihoodRatio(self, transformed_train_ivector, n,  # number of training utterances.
                           transformed_test_ivector):
        dim = self.Dim()

        mean = n * self.psi_ / (n * self.psi_ + 1.0) * transformed_train_ivector.squeeze()
        mean = mean.reshape(-1, 1)
        variance = 1.0 + self.psi_ / (n * self.psi_ + 1.0)
        logdet = np.sum(np.log(variance))
        sqdiff = transformed_test_ivector - mean
        sqdiff = np.power(sqdiff, 2)
        variance = 1 / variance
        loglike_given_class = -0.5 * (logdet + M_LOG_2PI * dim + np.matmul(sqdiff.T, variance.reshape(-1, 1)))
        # }
        # {// work out loglike_without_class.Here the mean is zero and the variance is I + \Psi.
        sqdiff = transformed_test_ivector  # there is no offset.
        sqdiff = np.power(sqdiff, 2)

        variance = self.psi_ + 1  # I + \Psi.
        logdet = np.sum(np.log(variance))
        variance = 1 / variance
        loglike_without_class = -0.5 * (logdet + M_LOG_2PI * dim + np.matmul(sqdiff.T, variance.reshape(-1, 1)))
        # }
        loglike_ratio = loglike_given_class - loglike_without_class

        return loglike_ratio.squeeze()

    def SmoothWithinClassCovariance(self, smoothing_factor):

        assert (smoothing_factor >= 0.0 and smoothing_factor <= 1.0)
        # smoothing_factor > 1.0 is possible but wouldn't really make sense.
        print("Smoothing within-class covariance by " + str(smoothing_factor) + ", Psi is initially: " + str(self.psi_))
        within_class_covar = np.ones(self.Dim())  # It's now the current within-class covariance
        # (a diagonal matrix) in the space transformed
        # by transform_.
        within_class_covar += smoothing_factor * self.psi_
        # We now revise our estimate of the within-class covariance to this
        # larger value.  This means that the transform has to change to as
        # to make this new, larger covariance unit.  And our between-class
        # covariance in this space is now less.

        self.psi_ /= within_class_covar
        print("New value of Psi is " + str(self.psi_))

        within_class_covar = 1 / np.sqrt(within_class_covar)
        self.transform_ = self.transform_ * within_class_covar
        self.ComputeDerivedVars()

    def ApplyTransform(self, in_transform):
        assert (len(in_transform) <= self.Dim() and in_transform.shape[1] == self.Dim())

        # Apply in_transform to mean_.
        mean_new = in_transform * self.mean_
        self.mean_ = mean_new

        transform_invert = self.transform_.copy()

        # Next, compute the between_var and within_var that existed
        # prior to diagonalization.
        psi_mat = np.diag(np.diag(self.psi_))
        transform_invert = 1 / transform_invert

        within_var = transform_invert
        between_var = transform_invert * psi_mat

        # Next, transform the variances using the input transformation.
        between_var_new = in_transform * between_var
        within_var_new = in_transform * within_var

        # Finally, we need to recompute psi_ and transform_. The remainder of
        # the code in this function  is a lightly modified copy of
        # PldaEstimator::GetOutput().
        transform1 = ComputeNormalizingTransform(within_var_new)

        # Now transform is a matrix that if we project with it, within_var becomes unit.
        # between_var_proj is between_var after projecting with transform1.
        between_var_proj = transform1 * between_var_new

        # Do symmetric eigenvalue decomposition between_var_proj = U diag(s) U^T,
        # where U is orthogonal.
        s, U = np.linalg.eig(between_var_proj)

        assert (s.min() >=0 )
        n = ApplyFloor(s, 0.0)

        if n>0:
            print("Floored " + str(n) + " eigenvalues of between-class variance to zero.")

        # Sort from greatest to smallest eigenvalue.
        sortindex_s = np.argsort(-s)
        s = s[sortindex_s]
        U = U[sortindex_s]

        # The transform U^T will make between_var_proj diagonal with value s
        # (i.e. U^T U diag(s) U U^T = diag(s)).  The final transform that
        # makes within_var unit and between_var diagonal is U^T transform1,
        # i.e. first transform1 and then U^T.
        self.transform_ = np.transpose(U) * transform1
        self.psi_ = s

        self.ComputeDerivedVars()

    def Read(self, plda_file):
        if not os.path.exists(plda_file):
            raise FileExistsError(plda_file)

        with open(plda_file, 'rb') as f:
            plda_header = f.read(9)
            assert (plda_header == b'\x00B<Plda> ')
            self.mean_ = _read_vec_flt_binary(f)
            self.transform_ = _read_mat_binary(f)
            self.psi_ = _read_vec_flt_binary(f)
            self.ComputeDerivedVars()
            plda_tailer = f.readline()
            assert (plda_tailer == b'</Plda> ')

    def Write(self, plda_file):
        if not os.path.exists(os.path.dirname(plda_file)):
            print('Making parent dir for plda files.')
            os.makedirs(os.path.dirname(plda_file))

        with open(plda_file, 'wb') as f:
            f.write(b'\x00B<Plda> ')

            write_vec_binary(f, self.mean_)
            write_mat_binary(f, self.transform_)
            write_vec_binary(f, self.psi_)

            f.write(b'</Plda> ')


class ClassInfo(object):
    def __init__(self, weight, mean, n):
        self.weight = weight
        self.mean = mean
        self.num_examples = n


class PldaStats(object):

    def __init__(self, dim):
        self.dim_ = dim
        self.num_classes_ = 0
        self.num_examples_ = 0  # total number of examples, summed over classes.
        self.class_weight_ = 0.0  # total over classes, of their weight.
        self.example_weight_ = 0.0  # total over classes, of weight times #examples.
        self.sum_ = np.zeros((dim, 1))
        # Weighted sum of class means (normalize by class_weight_ to get mean).
        # class means(normalize by
        # class_weight_ to get mean).
        # 使用weight计算的所有类的均值
        self.offset_scatter_ = np.zeros((dim, dim))
        # Sum over all examples, of the weight
                             # times (example - class-mean).
                             # 所有egs的weight加权均值

        self.class_info_ = []

    def Dim(self):
        return self.dim_

    def AddSamples(self, weight, group):
        # if self.num_examples_ == 0:
        #     self.dim_ = group.shape[-1]
        #     self.sum_ = np.zeros(self.dim_)
        #     self.offset_scatter_ = np.zeros((self.dim_, self.dim_))
        # else:
        assert (self.dim_ == group.shape[-1])

        n = len(group)
        mean = group.mean(axis=0)
        mean = mean.reshape((-1, 1))

        # mean->AddRowSumMat(1.0 / n, group);
        self.offset_scatter_ += weight * np.matmul(group.transpose(), group)

        # the following statement has the same effect as if we
        # had first subtracted the mean from each element of
        # the group before the statement above.
        self.offset_scatter_ += -n * weight * np.matmul(mean, mean.transpose())
        self.class_info_.append(ClassInfo(weight, mean, n))
        #
        self.num_classes_ += 1
        self.num_examples_ += n
        self.class_weight_ += weight
        self.example_weight_ += weight * n
        self.sum_ += weight * mean

    def is_sorted(self):
        for i in range(self.num_classes_ - 1):
            if self.class_info_[i].num_examples > self.class_info_[i + 1].num_examples:
                return False

        return True

    def sort(self):

        for i in range(self.num_classes_ - 1):
            for j in range(i + 1, self.num_classes_):
                if self.class_info_[i].num_examples > self.class_info_[j].num_examples:
                    self.class_info_[i], self.class_info_[j] = self.class_info_[j], self.class_info_[i]


def ComputeNormalizingTransform(covar):
    # 使用Cholesky分解covarance矩阵计算投影矩阵
    # covar = C C^T
    # proj = C^{-1}
    # The matrix that makes covar unit is C^{-1}, because
    # C^{-1} covar C^{-T} = C^{-1} C C^T C^{-T} = I.

    C = np.linalg.cholesky(covar)
    return np.linalg.inv(C)


class PldaEstimationConfig(object):
    def __init__(self, num_em_iters=10):
        self.num_em_iters = num_em_iters


class PldaEstimator(object):

    def __init__(self, pldastats):
        self.class_info_ = []
        self.stats_ = pldastats

        # 类内方差、类间方差
        self.within_var_ = None
        self.between_var_ = None
        # These stats are reset on each iteration.
        # 类内方差统计量
        self.within_var_stats_ = None
        # 计算类内方差的样本数
        self.within_var_count_ = 0  # count corresponding to within_var_stats_

        # 类间方差统计量
        self.between_var_stats_ = None
        # 计算类间方差的样本数
        self.between_var_count_ = 0  # count corresponding to within_var_stats_

        self.InitParameters()

        #   KALDI_DISALLOW_COPY_AND_ASSIGN(PldaEstimator);

    def InitParameters(self):
        self.within_var_ = np.identity(self.stats_.dim_)
        self.between_var_ = np.identity(self.stats_.dim_)

    def ComputeObjfPart1(self):
        # Returns the part of the obj relating to the class means (total_not normalized)
        # 计算类均值
        within_class_count = self.stats_.example_weight_ - self.stats_.class_weight_
        # within_logdet=0
        # det_sign=0

        #   SpMatrix<double> inv_within_var(within_var_);
        #   inv_within_var.Invert(&within_logdet, &det_sign);
        inv_within_var = np.linalg.inv(self.within_var_)
        within_logdet = np.linalg.det(self.within_var_)

        #   KALDI_ASSERT(det_sign == 1 && "Within-class covariance is singular");
        objf = -0.5 * (within_class_count * (within_logdet + M_LOG_2PI * self.Dim()) \
                       + np.trace(inv_within_var * self.stats_.offset_scatter_))
        return objf

    def ComputeObjfPart2(self):
        tot_objf = 0.0
        n = -1  # the number of examples for the current class

        combined_inv_var = np.linalg.inv(self.between_var_ + self.within_var_ / n)

        for i in range(len(self.stats_.class_info_)):
            info = self.stats_.class_info_[i]
            if (info.num_examples != n):
                n = info.num_examples;
                # variance of mean of n examples is between-class + 1/n * within-class
                combined_inv_var = 1.0 / n * self.within_var_
                combined_var_logdet = np.linalg.inv(combined_inv_var)

            mean = info.mean
            mean += -1.0 / self.stats_.class_weight_ * self.stats_.sum_
            tot_objf += info.weight * -0.5 * (combined_var_logdet + M_LOG_2PI * self.Dim() \
                                              + np.trace(mean * combined_inv_var * mean))
        return tot_objf;
        # Returns the objective-function per sample.
        # 计算每个egs的目标函数？

    def ComputeObjf(self):
        ans1 = self.ComputeObjfPart1()
        ans2 = self.ComputeObjfPart2()
        ans = ans1 + ans2
        example_weights = self.stats_.example_weight_
        normalized_ans = ans / example_weights;

        # print("Within-class objf per sample is %s between-class is %s, total is %s" % (
        # str(ans1 / example_weights), str(ans2 / example_weights), str(normalized_ans)))
        return normalized_ans

    def Dim(self):
        return self.stats_.Dim()

    # E-step
    def EstimateOneIter(self):
        self.ResetPerIterStats()
        # within_var_stats_ += offset_scatter_
        self.GetStatsFromIntraClass()  # 使用offset_scatter_更新within_var_stats_每个说话人的ivector类内方差
        # 使用每个spk的统计量更新更新between_var_stats_, within_var_stats_
        self.GetStatsFromClassMeans()  # 更新between_var_stats_, within_var_stats_
        # 更新within_var_, between_var_
        self.EstimateFromStats()

        # print()
        # print("Objective function is ", self.ComputeObjf())

    def ResetPerIterStats(self):
        self.within_var_stats_ = Resize((self.Dim(), self.Dim()))
        self.within_var_count_ = 0.0
        self.between_var_stats_ = Resize((self.Dim(), self.Dim()))
        self.between_var_count_ = 0.0

        # KALDI_LOG << "Trace of within-class variance is " << within_var_.Trace();
        # KALDI_LOG << "Trace of between-class variance is " << between_var_.Trace();
        print("Trace of within-class variance: %.4f between-class variance: %.4f" % (
        self.within_var_.trace(), self.between_var_.trace()))

    # gets stats from intra-class variation (stats_.offset_scatter_).
    def GetStatsFromIntraClass(self):
        # print(self.within_var_stats_.shape)
        # print(self.stats_.offset_scatter_.shape)

        self.within_var_stats_ += self.stats_.offset_scatter_
        # Note: in the normal case, the expression below will be equal to the sum
        # over the classes, of (1-n), where n is the #examples for that class.  That
        # is the rank of the scatter matrix that "offset_scatter_" has for that
        # class. [if weights other than 1.0 are used, it will be different.]
        self.within_var_count_ += (self.stats_.example_weight_ - self.stats_.class_weight_)

    # gets part of stats relating to class means.
    def GetStatsFromClassMeans(self):
        # SpMatrix<double> between_var_inv(between_var_);
        # between_var_inv.Invert();
        between_var_inv = np.linalg.inv(self.between_var_)

        #   SpMatrix<double> within_var_inv(within_var_);
        #   within_var_inv.Invert();
        within_var_inv = np.linalg.inv(self.within_var_)

        # mixed_var will equal (between_var^{-1} + n within_var^{-1})^{-1}.
        # mixed_var = np.array([]).reshape(0, self.Dim())
        # n = -1  # the current number of examples for the class.

        for i in range(len(self.stats_.class_info_)):

            info = self.stats_.class_info_[i]
            weight = info.weight
            if info.num_examples:
                n = info.num_examples
                mixed_var = between_var_inv + n * within_var_inv
                mixed_var = np.linalg.inv(mixed_var)  # todo

            m = info.mean + (-1.0 / self.stats_.class_weight_ * self.stats_.sum_)  # remove global mean
            m = m.reshape(-1, 1)
            temp = np.matmul(n * within_var_inv, m)
            w = np.matmul(mixed_var, temp)
            m_w = m - w
            m_w = m_w.reshape(-1, 1)

            self.between_var_stats_ += weight * mixed_var
            self.between_var_stats_ += weight * np.matmul(w, w.transpose())
            self.between_var_count_ += weight

            self.within_var_stats_ += weight * n * mixed_var
            self.within_var_stats_ += weight * n * np.matmul(m_w, m_w.transpose())
            self.within_var_count_ += weight

    # M-step
    def EstimateFromStats(self):
        self.within_var_ = (1. / self.within_var_count_) * self.within_var_stats_
        self.between_var_ = (1. / self.between_var_count_) * self.between_var_stats_

        # KALDI_LOG << "Trace of within-class variance is " << within_var_.Trace();
        # KALDI_LOG << "Trace of between-class variance is " << between_var_.Trace();

    def Estimate(self, config, plda):
        pbar = tqdm(range(config.num_em_iters))
        for i in pbar:
            pbar.set_description("Plda estimation iteration {:>2d} of {} ".format(i, config.num_em_iters))
            self.EstimateOneIter()
        self.GetOutput(plda)

    # Copy to output.
    def GetOutput(self, plda):
        plda.mean_ = 1. / self.stats_.class_weight_ * self.stats_.sum_

        print("Norm of mean of iVector distribution is ", np.linalg.norm(plda.mean_))
        # print("Norm of mean of iVector distribution is ", plda.mean_.Norm(2.0))
        # 计算使得within_var_对角化的变换矩阵transform1
        # within_var_ = C * C^T
        # transform1 = C^{-1}

        transform1 = ComputeNormalizingTransform(self.within_var_)
        # now transform is a matrix that if we project with it, within_var_ becomes unit.

        # between_var_proj是between_var做了transform1投影的矩阵
        between_var_proj = np.matmul(transform1, self.between_var_).__matmul__(transform1.transpose())

        # Do symmetric eigenvalue decomposition between_var_proj = U diag(s) U^T,
        # where U is orthogonal.
        # 分解between_var_proj得到有特征值的矩阵和对角化的投影矩阵U
        # between_var_proj = U diag(s) U^T
        s, U = np.linalg.eig(between_var_proj)

        assert (s.min() >= 0.0)
        n = ApplyFloor(s, 0.0)  # 特征值大于零，返回设置为0的数目n
        if (n > 0):
            print("Floored %d eigenvalues of between-class variance to zero." % n)

        # Sort from greatest to smallest eigenvalue.
        # 特征值和对应向量向量排序

        s_idx = np.flipud(np.argsort(s))
        s = s[s_idx]
        U = U.transpose()[s_idx].transpose()

        # The transform U^T will make between_var_proj diagonal with value s
        # (i.e. U^T U diag(s) U U^T = diag(s)).  The final transform that
        # makes within_var_ unit and between_var_ diagonal is U^T transform1,
        # i.e. first transform1 and then U^T.
        # print("s is: \n", s)
        # print("U is: \n", U)
        # print("transform1 is: \n", transform1)
        plda.transform_ = np.matmul(U.transpose(), transform1)
        plda.psi_ = s  # 更新psi

        # print(plda.transform_)
        # print(plda.psi_ )
        print("Diagonal of between-class variance in normalized space is:\n", s)
        plda.ComputeDerivedVars()

        # print("within var is: ", self.within_var_)
        # print("between var is: ", self.between_var_)
    #
