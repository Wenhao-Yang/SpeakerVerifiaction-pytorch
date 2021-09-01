#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: plt_det.py
@Time: 2020/11/1 17:36
@Overview:
"""
import argparse
import glob
import os

# Training settings
from eval_metrics import save_det

parser = argparse.ArgumentParser(description='PyTorch Speaker Recognition')
# Data options
parser.add_argument('--score-name', type=str, help='names for score files splited by ,')
parser.add_argument('--score-file', type=str, help='paths for score files splited by ,')
parser.add_argument('--save-path', type=str, help='paths for saving det plot')
parser.add_argument('--pf-max', default=0.1, type=float,
                    help='the maximum value for axis(default: 0.1(10%))')

args = parser.parse_args()

if __name__ == '__main__':

    score_files = list(args.score_file.split(','))
    score_names = list(args.score_name.split(','))

    assert len(score_files) == len(score_names)

    save_path = args.save_path
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    scores = []
    names = []
    for i in range(len(score_files)):
        if os.path.exists(score_files[i]):
            scores.append(score_files[i])
            names.append(score_names[i])

    if len(scores) > 0:
        save_det(save_path=save_path, score_files=scores, names=names, pf_max=args.pf_max)
        print("Saving det.png to %s !" % save_path)
    else:
        print("There is not score files in %s !" % args.score_file)
