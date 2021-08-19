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
parser.add_argument('--score-dir', type=str,
                    default='Data/xvector/LoResNet8/timit/spect_egs_None',
                    help='path to scp file for xvectors')
parser.add_argument('--save-path', type=str,
                    default='Data/xvector/LoResNet8/timit/spect_egs_None',
                    help='path to scp file for xvectors')
parser.add_argument('--pf-max', default=0.1, type=float,
                    help='num of speakers to plot (default: 10)')

args = parser.parse_args()

if __name__ == '__main__':

    loss_dir = args.score_dir
    score_files = glob.glob(loss_dir + "/*/scores")
    score_files.sort()
    save_path = args.save_path if os.path.exists(args.save_path) else args.score_dir

    if len(score_files) > 0:
        names = []
        for s in score_files:
            names.append(os.path.basename(os.path.dirname(s)))

        save_det(save_path=save_path, score_files=score_files, names=names, pf_max=args.pf_max)
        print("Saving det.png to %s !" % save_path)
    else:
        print("There is not score files in %s !" % loss_dir)
