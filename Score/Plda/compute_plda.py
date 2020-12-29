#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: compute_plda.py
@Time: 2020/12/22 21:28
@Overview: modified from "kaldi/src/ivector/ivector-compute-plda.cc"
"""

import argparse

import kaldi_io
import numpy as np

from Score.Plda.plda import PldaEstimationConfig, PldaStats, PldaEstimator, PLDA

# Training settings
parser = argparse.ArgumentParser(description='Kalid PLDA compute')
# Data options
parser.add_argument('--spk2utt', type=str, required=True, help='path to spk2utt')
parser.add_argument('--ivector-scp', type=str, required=True, help='path to ivector.scp')
parser.add_argument('--num-em-iters', type=int, default=10, help='path to ivector.scp')
parser.add_argument('--plda-file', type=str, required=True, help='path to plda directory')
parser.add_argument('--vector-format', type=str, default='kaldi', help='path to plda directory')

args = parser.parse_args()

# kaldi command line:
# ivector-compute-plda [options] <spk2utt-rspecifier> <ivector-rspecifier> <plda-out>


if __name__ == '__main__':
    # spk2utt_path = args.spk2utt
    # ivector_scp = args.ivector_scp
    # plda_dir = args.plda_dir
    if args.vector_format == 'kaldi':
        vec_loader = kaldi_io.read_vec_flt
    elif args.vector_format == 'npy':
        vec_loader = np.load
    else:
        raise ValueError(args.vector_format)

    plda_config = PldaEstimationConfig(num_em_iters=args.num_em_iters)

    num_spk_done = 0
    num_spk_err = 0
    num_utt_done = 0
    num_utt_err = 0
    vec_dim = -1

    utt2vec = {}
    with open(args.ivector_scp, 'r') as f:
        for l in f.readlines():
            uid, vec_path = l.split()
            utt2vec[uid] = vec_path
            # spks[spk_utts[0]] = spk_utts[1:]
            if vec_dim == -1:
                # vec_dim = vec_loader(os.path.join('Score/data', vec_path)).shape[-1] #Todo: change the dir
                vec_dim = vec_loader(vec_path).shape[-1]  # Todo: change the dir

    plda_stats = PldaStats(dim=vec_dim)  # 记录说话人数目及其egs数目、维度类中心等\
    spks = {}
    with open(args.spk2utt, 'r') as f:
        spk_err = []
        for l in f.readlines():
            spk_utts = l.split()
            this_vecs = []
            for uid in spk_utts[1:]:
                try:
                    # vec_path = os.path.join('Score/data', utt2vec[uid]) #Todo: change the dir
                    vec_path = utt2vec[uid]  # Todo: change the dir
                    this_vec = vec_loader(vec_path)
                    this_vecs.append(this_vec)
                    num_utt_done += 1
                except Exception as e:
                    num_utt_err += 1

            ivector_mat = np.array(this_vecs)
            weight = 1.0
            if len(ivector_mat) == 0:
                spk_err.append(spk_utts[0])
                num_spk_err += 1
            else:
                plda_stats.AddSamples(weight, ivector_mat)
                # spks[spk_utts[0]] = spk_utts[1:]
                num_spk_done += 1
        if len(spk_err) > 0:
            print("Not producing output for speaker: \n %s \nsince no utterances had iVectors" % spk_err)

    plda_stats.sort()  # spk信息类按照egs排序
    plda_estimator = PldaEstimator(plda_stats)  # 使用统计量类new一个训练估计参数的类
    plda = PLDA()
    plda_estimator.Estimate(plda_config, plda)
    plda.Write(args.plda_file)
    print('Writing plda model files in: %s' % args.plda_file)

    # WriteKaldiObject(plda, args.plda_dir, binary)
    # return num_spk_done!=0 &, 1
