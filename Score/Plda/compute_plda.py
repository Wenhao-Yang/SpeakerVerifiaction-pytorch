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
import os

import kaldi_io
import numpy as np
from kaldi_io.kaldi_io import _read_vec_flt_binary, UnknownVectorHeader
from tqdm import tqdm

from Score.Plda.plda import PldaEstimationConfig, PldaStats, PldaEstimator, PLDA, write_vec_binary

# Training settings
parser = argparse.ArgumentParser(description='Kalid PLDA compute')
# Data options
parser.add_argument('--spk2utt', type=str, required=True, help='path to spk2utt')
parser.add_argument('--ivector-scp', type=str, required=True, help='path to ivector.scp')
parser.add_argument('--num-em-iters', type=int, default=10, help='path to ivector.scp')
parser.add_argument('--plda-file', type=str, required=True, help='path to plda directory')
parser.add_argument('--subtract-global-mean', action='store_false', default=True, help='path to plda directory')
parser.add_argument('--mean-vec', type=str, default='', help='path to plda directory')
parser.add_argument('--transform-vec', type=str, default='', help='path to plda directory')
parser.add_argument('--normalize-length', action='store_false', default=True, help='path to plda directory')
parser.add_argument('--scaleup', action='store_false', default=True, help='path to plda directory')
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
                vec_dim = vec_loader(os.path.join('Score/data', vec_path)).shape[-1] #Todo: change the dir
                # vec_dim = vec_loader(vec_path).shape[-1]  # Todo: change the dir


    spks = {}
    if args.subtract_global_mean:
        if args.mean_vec != "" and os.path.exists(args.mean_vec):
            with open(args.mean_vec, 'rb') as f:
                try:
                    global_mean = _read_vec_flt_binary(f)
                except UnknownVectorHeader as u:
                    mean_vec = []
                    vec_str = f.readline()
                    for v in vec_str.split():
                        try:
                            mean_vec.append(float(v))
                        except:
                            pass
                    global_mean = np.array(mean_vec)
        else:
            global_mean = []
            with open(args.spk2utt, 'r') as f:
                pbar = tqdm(f.readlines())
                for l in pbar:
                    spk_utts = l.split()
                    for uid in spk_utts[1:]:
                        try:
                            vec_path = os.path.join('Score/data', utt2vec[uid]) #Todo: change the dir
                            # vec_path = utt2vec[uid]  # Todo: change the dir
                            this_vec = vec_loader(vec_path)
                            global_mean.append(this_vec)
                        except Exception as e:
                            pass
            global_mean = np.array(global_mean).mean(axis=0)

            assert args.mean_vec != "", print("mean vector path should be assigned!")
            if not os.path.exists(os.path.dirname(args.mean_vec)):
                os.makedirs(os.path.dirname(args.mean_vec))
            with open(args.mean_vec, 'wb') as f:
                write_vec_binary(global_mean)
                print("Saving mean vector to: " % args.mean_vec)

    transform_vec = None
    if os.path.exists(args.transform_vec):
        try:
            with open(args.transform_vec, 'rb') as f:
                transform_vec = kaldi_io.read_mat(f)
                if transform_vec.shape[-1] != vec_dim:
                    transform_vec = transform_vec[:, :vec_dim]
                    vec_dim = transform_vec.shape[0]
                print("Transformed dim will be %d" % vec_dim)
        except Exception as e:
            print("Skippinng transform ... Transform vector loading error: \n%s" % str(e))

    tot_ratio = 0.0
    tot_ratio2 = 0.0

    plda_stats = PldaStats(dim=vec_dim)  # 记录说话人数目及其egs数目、维度类中心等
    with open(args.spk2utt, 'r') as f:
        spk_err = []
        for l in f.readlines():
            spk_utts = l.split()
            this_vecs = []
            for uid in spk_utts[1:]:
                try:
                    vec_path = os.path.join('Score/data', utt2vec[uid]) #Todo: change the dir
                    # vec_path = utt2vec[uid]  # Todo: change the dir
                    this_vec = vec_loader(vec_path)
                    this_vecs.append(this_vec)
                    num_utt_done += 1
                except Exception as e:
                    num_utt_err += 1

            ivector_mat = np.array(this_vecs)
            if args.subtract_global_mean:
                ivector_mat -= global_mean

            if transform_vec.all() != None:
                ivector_mat = np.matmul(ivector_mat, transform_vec.transpose())

            if args.normalize_length:
                norm = np.linalg.norm(ivector_mat, axis=1).reshape(-1, 1)
                ratio = norm / np.sqrt(ivector_mat.shape[-1]) if args.scaleup else norm

                assert ratio.all()>0.0

                ivector_mat /= ratio
                tot_ratio += ratio.sum()
                tot_ratio2 += (ratio * ratio).sum()

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

    # if (num_done != 0) {
    #       BaseFloat avg_ratio = tot_ratio / num_done,
    #           ratio_stddev = sqrt(tot_ratio2 / num_done - avg_ratio * avg_ratio);
    #       KALDI_LOG << "Average ratio of iVector to expected length was "
    #                 << avg_ratio << ", standard deviation was " << ratio_stddev;
    #     }
    if args.normalize_length:
        avg_ratio = tot_ratio / num_utt_done
        ratio_stddev = np.sqrt(tot_ratio2 / num_utt_done - avg_ratio * avg_ratio)
        print("Average ratio of iVector to expected length was ", avg_ratio, ", standard deviation was ", ratio_stddev)

    plda_stats.sort()  # spk信息类按照egs排序
    plda_estimator = PldaEstimator(plda_stats)  # 使用统计量类new一个训练估计参数的类
    plda = PLDA()
    plda_estimator.Estimate(plda_config, plda)
    plda.Write(args.plda_file)
    print('Writing plda model files in: %s' % args.plda_file)

    # WriteKaldiObject(plda, args.plda_dir, binary)
    # return num_spk_done!=0 &, 1

"""
ivector-compute-plda ark:/home/yangwenhao/local/project/lstm_speaker_verification/data/vox1/pyfb_de/dev_fb24_mel/spk2utt \
      "ark:ivector-subtract-global-mean scp:exp/ivectors_train_fb24_mel/ivector.scp ark:- | transform-vec exp/ivectors_train_fb24_mel/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
      exp/ivectors_train_fb24_mel/plda
      
"""
