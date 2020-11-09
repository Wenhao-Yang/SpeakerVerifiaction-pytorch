#!/usr/bin/env bash


stage=10
if [ $stage -le 0 ]; then

  python Vector_Score/enroll_eval.py \
    --data-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/army/spect/thre_enrolled \
    --extract-path Data/xvector/LoResNet10/army/spect_81/soft/x_vector \
    --test-dir Data/xvector/LoResNet10/army/spect_81/soft/x_vector \
    --split-set


fi

if [ $stage -le 10 ]; then

  python Vector_Score/enroll_eval.py \
    --data-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/army/spect/thre_enrolled \
    --extract-path Data/xvector/MultiResNet10/army/spect_81/soft/x_vector/enrollled \
    --test-dir Data/xvector/LoResNet10/army/spect_81/soft/x_vector \
    --split-set

fi