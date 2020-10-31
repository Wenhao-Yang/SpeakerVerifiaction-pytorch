#!/usr/bin/env bash


stage=20
if [ $stage -le 0 ]; then

  python Vector_Score/enroll_eval.py \
    --data-dir \
    --enroll-dir Data/xvector/LoResNet10/army/spect_81/soft/x_vector \
    --test-dir Data/xvector/LoResNet10/army/spect_81/soft/x_vector \
    --split-set


fi