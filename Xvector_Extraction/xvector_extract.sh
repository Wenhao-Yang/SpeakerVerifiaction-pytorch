#!/usr/bin/env bash

stage=20
if [ $stage -le 0 ]; then
  model=ASTDNN
  feat=mfcc40
  loss=soft
  python Xvector_Extraction/extract_xvector_kaldi.py \
    --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pymfcc40/dev_kaldi \
    --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pymfcc40/test_kaldi \
    --check-path Data/checkpoint/${model}/${feat}/${loss} \
    --resume Data/checkpoint/${model}/${feat}/${loss}/checkpoint_20.pth \
    --sitw-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/sitw \
    --feat-dim 40 \
    --extract-path Data/xvector/${model}/${feat}/${loss} \
    --model ${model} \
    --dropout-p 0.0\
    --epoch 20 \
    --embedding-size 512
fi

if [ $stage -le 1 ]; then
  model=LoResNet10
  feat=spect_161
  loss=soft
  python Xvector_Extraction/extract_xvector_kaldi.py \
    --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pymfcc40/dev_kaldi \
    --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pymfcc40/test_kaldi \
    --check-path Data/checkpoint/${model}/${feat}/${loss} \
    --resume Data/checkpoint/${model}/${feat}/${loss}/checkpoint_20.pth \
    --sitw-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/sitw \
    --feat-dim 161 \
    --extract-path Data/xvector/${model}/${feat}/${loss} \
    --model ${model} \
    --dropout-p 0.0 \
    --epoch 20 \
    --embedding-size 1024
fi

if [ $stage -le 2 ]; then
  model=LoResNet10
  dataset=timit
  feat=spect_161
  loss=soft

  python Xvector_Extraction/extract_xvector_kaldi.py \
    --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/train_spect_noc \
    --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/test_spect_noc \
    --resume Data/checkpoint/LoResNet10/timit_spect/soft_fix/checkpoint_15.pth \
    --sitw-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/sitw \
    --feat-dim 161 \
    --embedding-size 128 \
    --extract-path Data/xvector/${model}/${dataset}/${feat}/${loss} \
    --model ${model} \
    --dropout-p 0.0 \
    --epoch 20 \
    --embedding-size 1024
fi

if [ $stage -le 20 ]; then
  model=LoResNet10
  dataset=army
  feat=spect_81
  loss=soft

  python Xvector_Extraction/extract_xvector_kaldi.py \
    --train-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/army/spect/dev_8k \
    --test-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/army/spect/thre_enrolled \
    --resume Data/checkpoint/LoResNet10/army_v1/spect_egs_fast_None/soft_dp01/checkpoint_20.pth \
    --feat-dim 81 \
    --embedding-size 128 \
    --fast \
    --time-dim 1 \
    --stride 1 \
    --dropout-p 0.1 \
    --channels 32,64,128,256 \
    --alpha 12.0 \
    --input-norm Mean \
    --encoder-type None \
    --extract-path Data/xvector/${model}/${dataset}/${feat}/${loss} \
    --model ${model} \
    --dropout-p 0.0 \
    --epoch 20 \
    --embedding-size 1024
fi
