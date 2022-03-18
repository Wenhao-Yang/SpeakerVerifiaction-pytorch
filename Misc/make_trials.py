#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: make_trials.py
@Time: 2022/3/18 21:52
@Overview:
"""
# import os
import pathlib
import random
import sys
# from tqdm import tqdm
import numpy as np

random.seed(123456)
np.random.seed(123456)

assert sys.argv[1].isdigit()
num_pair = int(sys.argv[1])
enroll_roots = sys.argv[2]
test_roots = sys.argv[3]
trials_path = sys.argv[4]

# print('Current path: ' + os.getcwd())
print("Enroll Dirs is: " + enroll_roots)
print("Test Dirs is: " + test_roots)

enroll_roots = pathlib.Path(enroll_roots)
test_roots = pathlib.Path(test_roots)

enroll_wavs = [w for w in enroll_roots.glob('*.wav')]
test_wavs = [w for w in test_roots.glob('*.wav')]

assert len(enroll_wavs) > 0, print('no wavs in %s' % (str(enroll_roots)))
assert len(test_wavs) > 0, print('no wavs in %s' % (str(test_roots)))

enroll_dict = {}

for w in enroll_wavs:
    sid, _ = str(w).split('/')[-1].split('-')  # 'enroll/id00884-enroll.wav'
    enroll_dict[sid] = str(w)

test_dict = {}
test_utt2spk_dict = {}

numofutts = 0

for w in test_wavs:
    sid, _, _, _ = str(w).split('/')[-1].split('-')  # 'enroll/id00884-enroll.wav'
    if sid in test_dict:
        test_dict[sid].append(str(w))
    else:
        test_dict[sid] = [str(w)]
    numofutts += 1

enroll_spks = list(enroll_dict.keys())

with open(trials_path, 'w') as f:
    trials = []
    utts = numofutts

    print('Num of repeats: %d ' % (num_pair / len(enroll_spks)))
    repeats = int(num_pair / len(enroll_spks))

    pairs = 0
    positive_pairs = set()
    negative_pairs = set()

    for sid in enroll_spks:

        spk_posi = 0
        this_spk_pairs = 0
        if sid in test_dict:
            for uid in test_dict[sid]:
                positive_pairs.add(' '.join((enroll_dict[sid], uid, 'target\n')))
                this_spk_pairs += 1
                if this_spk_pairs > 0.5 * repeats:
                    break

        if sid in test_dict:
            negative_per_spk = int((repeats - this_spk_pairs) / (len(test_dict) - 1))
        else:
            negative_per_spk = int((repeats - this_spk_pairs) / len(test_dict))

        for ne_sid in test_dict:
            i = 0
            if sid == ne_sid:
                continue
            else:
                for uid in test_dict[ne_sid]:
                    negative_pairs.add(' '.join((enroll_dict[sid], uid, 'nontarget\n')))
                    i += 1
                    if i > negative_per_spk:
                        break

    positive_pairs = list(positive_pairs)
    negative_pairs = list(negative_pairs)
    # pdb.set_trace()

    random.shuffle(negative_pairs)
    random.shuffle(positive_pairs)

    num_positive = len(positive_pairs)
    for l in negative_pairs:
        positive_pairs.append(l)

    random.shuffle(positive_pairs)
    num_pair = len(positive_pairs)
    for l in positive_pairs:
        f.write(l)

    print('Generate %d pairs for set: %s, in which %d of them are positive pairs.' % (
        num_pair, trials_path, num_positive))
