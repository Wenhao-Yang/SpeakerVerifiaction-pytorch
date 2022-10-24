#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: filter_vectors.py
@Time: 2022/10/24 17:09
@Overview:
"""
import argparse
import glob
import os
import numpy as np
# Training settings
from eval_metrics import save_det

parser = argparse.ArgumentParser(description='PyTorch Speaker Recognition')
# Data options
parser.add_argument('--score-file', type=str, help='paths for score files splited by ,')
parser.add_argument('--trials', type=str, help='paths for score files splited by ,')
parser.add_argument('--threshold', type=float, help='paths for saving det plot')
parser.add_argument('--output-file', default='', help='the maximum value for axis(default: 0.1(10%))')
parser.add_argument('--confidence-interval', type=float, default=0.1, help='paths for saving det plot')

args = parser.parse_args()

Confidence_interval = args.confidence_interval
threshold = args.threshold

pairs = []
with open(args.trials, 'r') as f:
    for l in f.readlines():
        a, b, truth = l.split()
        pairs.append([a, b])

ambigous_uids = set([])
with open(args.score_file, 'r') as f:
    i = 0
    for l in f.readlines():
        result, score = l.split()
        score = float(score)

        if result in ['True', '1']:
            if score - threshold <= Confidence_interval:
                ambigous_uids.add(a)
                ambigous_uids.add(b)
        else:
            if threshold - score <= Confidence_interval:
                ambigous_uids.add(a)
                ambigous_uids.add(b)

with open(args.output_file, 'w') as f:
    for uid in list(ambigous_uids):
        f.write(uid + '\n')

print('Saved %d utterances with low confidence.' % len(ambigous_uids))
