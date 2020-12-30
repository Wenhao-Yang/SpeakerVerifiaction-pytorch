#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: compute_lda.py
@Time: 2020/12/29 19:35
@Overview:
"""
import argparse
import os

import kaldi_io
import numpy as np

from Score.Plda.lda import ComputeAndSubtractMean, ComputeLdaTransform
from Score.Plda.plda import write_mat_binary

# Training settings
parser = argparse.ArgumentParser(description='Kalid PLDA compute')
# Data options
parser.add_argument('--spk2utt', type=str, required=True, help='path to spk2utt')
parser.add_argument('--ivector-scp', type=str, required=True, help='path to ivector.scp')
parser.add_argument('--lda-mat', type=str, required=True, help='path to plda directory')
parser.add_argument('--vector-format', type=str, default='kaldi', help='path to plda directory')

parser.add_argument('--lda-dim', type=int, default=100, help='path to spk2utt')
parser.add_argument('--total-covariance-factor', type=float, default=0.0, help='path to spk2utt')
parser.add_argument('--covariance-floor', type=float, default=1e-6, help='path to spk2utt')

args = parser.parse_args()

# kaldi command line:
# ivector-compute-lda [options] <ivector-rspecifier> <utt2spk-rspecifier> <lda-matrix-out>


if __name__ == '__main__':
    # 默认lda后为100维
    lda_dim = args.lda_dim  # Dimension we reduce to
    total_covariance_factor = args.total_covariance_factor
    covariance_floor = args.covariance_floor

    if args.vector_format == 'kaldi':
        vec_loader = kaldi_io.read_vec_flt
    elif args.vector_format == 'npy':
        vec_loader = np.load
    else:
        raise ValueError(args.vector_format)

    assert (covariance_floor >= 0.0);

    num_done = 0
    num_err = 0
    dim = 0

    utt2vec = {}
    with open(args.ivector_scp, 'r') as f:
        for l in f.readlines():
            try:
                uid, vec_path = l.split()
                # this_vec = vec_loader(vec_path)
                this_vec = vec_loader(os.path.join('Score/data', vec_path))
                utt2vec[uid] = this_vec

                if dim == 0:
                    # vec_dim = vec_loader(os.path.join('Score/data', vec_path)).shape[-1] #Todo: change the dir
                    dim = this_vec.shape[-1]  # Todo: change the dir
                else:
                    assert (dim == this_vec.shape[-1])
                num_done += 1
            except Exception as e:
                num_err += 1

    spk2utt = {}
    with open(args.spk2utt, 'r') as f:
        spk_err = []
        for l in f.readlines():
            spk_utts = l.split()
            spk = spk_utts[0]
            spk2utt[spk] = spk_utts[1:]

    print("Read %d utterances, %d with errors." % (num_done, num_err))

    if num_done == 0:
        raise Exception("Did not read any utterances.")
    else:
        print("Computing within-class covariance.")

    # 计算ivector的均值
    mean = ComputeAndSubtractMean(utt2vec)
    print("2-norm of iVector mean is %f " % np.sqrt(np.power(mean, 2).sum()))

    # LDA matrix without the offset term.
    lda_mat = np.zeros((lda_dim, dim + 1))
    # 初始化linear_part = lda_mat[0:lda_dim][0:dim]
    linear_part = lda_mat[0:lda_dim, 0:dim].copy()
    # 计算变换的矩阵linear_part
    ComputeLdaTransform(utt2vec, spk2utt, total_covariance_factor, covariance_floor, linear_part)

    # y = -1 * linear_part + 0 * mean.
    offset = -1.0 * np.matmul(linear_part, mean.reshape(-1, 1))
    lda_mat[:, dim] = offset.squeeze()  # add mean-offset to transform
    # 把offset加到lda_mat
    print("2-norm of transformed iVector mean is ", np.power(offset, 2.0).sum())

    lda_file = args.lda_mat
    if not os.path.exists(os.path.dirname(lda_file)):
        print('Making parent dir for lda files.')
        os.makedirs(os.path.dirname(lda_file))

    with open(lda_file, 'wb') as f:
        write_mat_binary(f, lda_mat)

    print("Wrote LDA transform mat to ", lda_mat)
