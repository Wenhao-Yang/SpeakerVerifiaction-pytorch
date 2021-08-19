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

from Score.Plda.lda import ComputeAndSubtractMean, ComputeLdaTransform, SubtractGlobalMean
from Score.Plda.plda import write_mat_binary

# Training settings
parser = argparse.ArgumentParser(description='Kalid PLDA compute')
# Data options
parser.add_argument('--spk2utt', type=str, required=True, help='path to spk2utt')
parser.add_argument('--ivector-scp', type=str, required=True, help='path to ivector.scp')
parser.add_argument('--subtract-global-mean', action='store_true', default=True,
                    help='ivector subtract global mean while reading')

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

    utt2vec_path = {}
    with open(args.ivector_scp, 'r') as f:
        for l in f.readlines():
            try:
                uid, vec_path = l.split()
                utt2vec_path[uid] = vec_path
            except:
                continue

    utt2vec = {}
    spk2utt = {}
    with open(args.spk2utt, 'r') as f:
        spk_err = []
        for l in f.readlines():
            spk_utts = l.split()
            spk = spk_utts[0]
            spk2utt[spk] = spk_utts[1:]

            for utt in spk_utts[1:]:
                try:
                    vec_path = utt2vec_path[utt]
                    this_vec = vec_loader(os.path.join('Score/data', vec_path))
                    utt2vec[utt] = this_vec
                    if dim == 0:
                        # vec_dim = vec_loader(os.path.join('Score/data', vec_path)).shape[-1] #Todo: change the dir
                        # print('Dtype of ivectors is ', this_vec.dtype)

                        dim = this_vec.shape[-1]  # Todo: change the dir
                    else:
                        assert (dim == this_vec.shape[-1])

                    num_done += 1
                except Exception as e:
                    num_err += 1

    print("Read %d utterances, %d with errors." % (num_done, num_err))

    if num_done == 0:
        raise Exception("Did not read any utterances.")
    else:
        print("Computing within-class covariance.")

    # 计算ivector的均值
    if args.subtract_global_mean:
        SubtractGlobalMean(utt2vec)

    mean = ComputeAndSubtractMean(utt2vec)
    # print("mean vector is ", str(mean))
    print("2-norm of iVector mean is %f " % np.linalg.norm(mean))

    # LDA matrix without the offset term.
    # 初始化linear_part
    linear_part = np.zeros((lda_dim, dim))
    # 计算变换的矩阵linear_part
    linear_part = ComputeLdaTransform(utt2vec, spk2utt, total_covariance_factor, covariance_floor, linear_part)
    # print("linear_part matrix: ", linear_part)

    # y = -1 * linear_part * mean.
    offset = -1.0 * np.matmul(linear_part, mean.reshape(-1, 1))

    # 把offset加到lda_mat
    lda_mat = np.concatenate((linear_part, offset), axis=1)  # add mean-offset to transform

    print("2-norm of transformed iVector mean is ", np.sqrt(np.power(offset, 2.0).sum()))
    lda_file = args.lda_mat
    if not os.path.exists(os.path.dirname(lda_file)):
        print('Making parent dir for lda files.')
        os.makedirs(os.path.dirname(lda_file))

    # print("LDA matrix: ", lda_mat)
    with open(lda_file, 'wb') as f:
        write_mat_binary(f, lda_mat)

    print("Wrote LDA transform mat to ", lda_file)

"""
ivector-compute-lda --total-covariance-factor=0.0 --dim=100 \
      "ark:ivector-subtract-global-mean scp:exp/ivectors_train_fb24_mel/ivector.scp ark:- |" \
      ark:/home/yangwenhao/local/project/lstm_speaker_verification/data/vox1/pyfb_de/dev_fb24_mel/utt2spk exp/ivectors_train_fb24_mel/transform.mat
"""
