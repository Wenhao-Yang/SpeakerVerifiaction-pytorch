#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: plda_scoring.py
@Time: 2020/12/24 16:16
@Overview: modified from "kaldi/src/ivector/ivector-plda-scoring.cc"
"""
import argparse
import os

import kaldi_io
import numpy as np

from Vector_Score.plda import PLDA, PldaConfig

# Training settings
parser = argparse.ArgumentParser(description='Kalid PLDA scoring')
# Data options
parser.add_argument('--train-vec-scp', type=str, required=True, help='path to transformed spk_ivector.scp')
parser.add_argument('--test-vec-scp', type=str, help='path to transformed test_ivector.scp')
parser.add_argument('--trials', type=str, required=True, help='path to trials')
parser.add_argument('--plda-file', type=str, required=True, help='path to plda')
parser.add_argument('--score', type=str, required=True, help='path to scores file')

parser.add_argument('--vector-format', type=str, default='kaldi', help='path to plda directory')
parser.add_argument('--num-utts', type=str, default="", help='path to plda directory')

parser.add_argument('--normalize-length', action='store_true', default=True, help='log power spectogram')
parser.add_argument('--simple-length-norm', action='store_true', default=False, help='path to plda directory')

args = parser.parse_args()

# kaldi command lines:
# ivector-plda-scoring <plda> <train-ivector-rspecifier> <test-ivector-rspecifier> <trials-rxfilename> <scores-wxfilename>

if __name__ == '__main__':
    tot_test_renorm_scale = 0.0
    tot_train_renorm_scale = 0.0
    num_train_ivectors = 0
    num_train_errs = 0

    num_test_ivectors = 0

    num_trials_done = 0
    num_trials_err = 0

    plda_config = PldaConfig()
    plda_config.register(normalize_length=args.normalize_length,
                         simple_length_norm=args.simple_length_norm)

    plda = PLDA()
    plda.Read(args.plda_file)

    dim = plda.Dim()

    spk2num_utts = {}
    if args.num_utts != '':
        if not os.path.exists(args.num_utts):
            raise FileExistsError(args.num_utts)
        else:
            with kaldi_io.open_or_fd(args.num_utts) as f:
                while True:
                    key = kaldi_io.read_key(f)
                    if key != "":
                        value = kaldi_io.read_vec_int(f)
                        spk2num_utts[key] = value[0]
                    else:
                        break

    print('Reading train iVectors:')
    train_ivectors = {}
    for spk, ivector in kaldi_io.read_vec_flt_scp(args.train_vec_scp):
        # train_vec[spk] = ivector

        if len(spk2num_utts) > 0:
            if spk in spk2num_utts:
                num_examples = spk2num_utts[spk]
            else:
                num_train_errs += 1
                continue
        else:
            num_examples = 1

        transformed_ivector, normalization_factor = plda.TransformIvector(plda_config, ivector, num_examples)
        # print(transformed_ivector.shape)

        tot_train_renorm_scale += normalization_factor
        train_ivectors[spk] = transformed_ivector

        num_train_ivectors += 1

    if (num_train_ivectors == 0):
        print("No training iVectors present.")
    print("Average renormalization scale on %d training iVectors was %.4f" % (
    num_train_ivectors, tot_train_renorm_scale / num_train_ivectors))

    print('Reading test iVectors:')
    test_ivectors = {}
    for uid, ivector in kaldi_io.read_vec_flt_scp(args.test_vec_scp):
        # train_vec[spk] = ivector
        if uid in test_ivectors:
            print("Duplicate test iVector found for utterance %s" % uid)
        num_examples = 1

        transformed_ivector, normalization_factor = plda.TransformIvector(plda_config, ivector, num_examples)

        tot_test_renorm_scale += normalization_factor
        test_ivectors[uid] = transformed_ivector

        num_test_ivectors += 1

    if (num_test_ivectors == 0):
        print("No testing iVectors present.")

    print("Average renormalization scale on %d testing iVectors was %.4f" % (
    num_test_ivectors, tot_test_renorm_scale / num_test_ivectors))

    if not os.path.exists(os.path.dirname(args.score)):
        print("Making score dir...")
        os.makedirs(os.path.dirname(args.score))

    score_sum = 0.0
    sumsq = 0.0

    if not os.path.exists(args.trials):
        raise FileExistsError(args.trials)
    else:
        with open(args.trials, 'r') as f, open(args.score, 'w') as s:
            for l in f.readlines():
                key1, key2, _ = l.split()

                if key1 not in train_ivectors:
                    print("Key ", key1, " not present in training iVectors.")
                    num_trials_err += 1

                if key2 not in test_ivectors:
                    print("Key ", key2, " not present in testing iVectors.")
                    num_trials_err += 1

                train_ivector = train_ivectors[key1]
                test_ivector = test_ivectors[key2]

                if len(spk2num_utts) > 0:
                    num_train_examples = spk2num_utts[key1]
                else:
                    num_train_examples = 1

                score = plda.LogLikelihoodRatio(train_ivector,
                                                num_train_examples,
                                                test_ivector)

                score_sum += score
                sumsq += score * score
                num_trials_done += 1
                s.write(' '.join([key1, key2, str(score) + '\n']))

    if (num_trials_done != 0):
        mean = score_sum / num_trials_done
        scatter = sumsq / num_trials_done
        variance = scatter - mean * mean
        stddev = np.sqrt(variance)

        print("Mean score was %.4f, standard deviation was %.4f" % (mean, stddev))

    print("Processed %d trials, %d had errors." % (num_trials_done, num_trials_err))
