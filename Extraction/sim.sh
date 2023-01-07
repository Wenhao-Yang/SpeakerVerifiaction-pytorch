#!/usr/bin/env bash

stage=0
lstm_dir=/home/yangwenhao/project/lstm_speaker_verification

# ===============================    LoResNet10    ===============================
if [ $stage -le 0 ]; then
  for epoch in 10; do
    test_set=cnceleb subset=dev test_input=fix

    model_dir=ThinResNet34/cnceleb/klfb_egs_baseline/arcsoft_sgd_rop/Mean_batch256_seblock_red2_downk1_avg5_ASTP2_em256_dp01_alpha0_none1_wde4_varesmix2_bashuf2_dist/123456
    data_dir=Data/xvector/${model_dir}/${test_set}_${subset}_${epoch}_${test_input}/test
    
    python Extraction/compute_dist.py \
        --data-dir ${data_dir} \
        --checkpoint Data/checkpoint/${model_dir}/checkpoint_${epoch}.pth \
        --gpu-id 0
  done
fi