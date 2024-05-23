#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: ScoreCohort.py
@Time: 2022/3/26 18:34
@Overview:
"""
import random

import argparse

# Training settings
import kaldiio
import numpy as np
import torch

parser = argparse.ArgumentParser(description='Score vectors for training set')
# Model options
parser.add_argument('--train-scp', type=str, help='path to dataset')
parser.add_argument('--trials', type=str, default='trials', help='path to voxceleb1 test dataset')

parser.add_argument('--enroll-scp', type=str, help='path to voxceleb1 test dataset')
parser.add_argument('--test-scp', type=str, help='path to voxceleb1 test dataset')

parser.add_argument('--out-scp', type=str, help='path to voxceleb1 test dataset')
parser.add_argument('--cohort-size', type=int, default=20000,
                    help='how many batches to wait before logging training status')
parser.add_argument('--n-train-snts', type=int, default=400000,
                    help='how many batches to wait before logging training status')
parser.add_argument('--data-format', type=str, default='kaldi', help='path to voxceleb1 test dataset')


# Device options
parser.add_argument('--seed', type=int, default=1234, metavar='S',
                    help='random seed (default: 0)')

args = parser.parse_args()


vector_loader = kaldiio.load_mat if args.data_format == 'kaldi' else np.load

Similarity = torch.nn.CosineSimilarity()

train_vectors = []
train_scps = []
with open(args.train_scp, 'r') as f:
    for l in f.readlines():
        uid, vpath = l.split()
        train_scps.append((uid, vpath))

random.shuffle(train_scps)

if args.n_train_snts > len(train_scps):
    train_scps = train_scps[:train_scps]

for (uid, vpath) in train_scps:

    train_vectors.append(vector_loader(vpath))


train_vectors = torch.tensor(train_vectors)

test_uid2vector = {}
test_scps = set()
# eval_scps = set()

with open(args.trials, 'r') as f:
    for l in f.readlines():
        uid_enroll, uid_eval, _  = l.split()
        test_scps.add(uid_enroll)
        test_scps.add(uid_eval)

        # train_scps.append((uid, vpath))


test_uid2vector = {}
with open(args.test_scp, 'r') as f:
    for l in f.readlines():
        uid, vpath = l.split()

        test_uid2vector[uid] = vector_loader(vpath)

test_stats = {}


for uid in test_scps:

    test_vector = torch.tensor(test_uid2vector[uid])
    test_vector = test_vector.repeat(train_vectors.shape[0], 1)

    scores = Similarity(test_vector, train_vectors)
    scores = torch.topk(scores, k=args.cohort_size, dim=0)[0]

    mean_t_c = torch.mean(scores, dim=0)
    std_t_c = torch.std(scores, dim=0)

    test_stats[uid] = [mean_t_c, std_t_c]
