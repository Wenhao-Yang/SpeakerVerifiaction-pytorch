#!/usr/bin/env bash

stage=40
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
  model=LoResNet
  dataset=army
  feat=spect_81
  loss=soft
  resnet_size=10

  python Xvector_Extraction/extract_xvector_kaldi.py \
    --train-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/army/spect/dev_8k \
    --test-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/army/spect/thre_enrolled \
    --resume Data/checkpoint/LoResNet10/army_v1/spect_egs_fast_None/soft_dp01/checkpoint_20.pth \
    --feat-dim 81 \
    --train-spk 3162 \
    --embedding-size 128 \
    --batch-size 1 \
    --fast \
    --time-dim 1 \
    --stride 1 \
    --dropout-p 0.1 \
    --channels 32,64,128,256 \
    --alpha 12.0 \
    --input-norm Mean \
    --encoder-type None \
    --extract-path Data/xvector/${model}${resnet_size}/${dataset}/${feat}/${loss} \
    --model ${model} \
    --resnet-size ${resnet_size} \
    --dropout-p 0.0 \
    --epoch 20
fi

if [ $stage -le 40 ]; then
  model=MultiResNet
  dataset=army
  feat=spect_81
  loss=soft
  resnet_size=10

  python Xvector_Extraction/extract_xvector_multi.py \
    --enroll-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/army/spect/thre_enrolled \
    --test-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/army/spect/thre_notenrolled \
    --resume Data/checkpoint/MultiResNet10/army_x2/spect_egs_None/center_dp25_b192_16_0.01/checkpoint_24.pth \
    --feat-dim 81 \
    --train-spk-a 1951 \
    --train-spk-b 1211 \
    --embedding-size 128 \
    --batch-size 64 \
    --time-dim 1 \
    --avg-size 4 \
    --stride 1 \
    --channels 16,64,128,256 \
    --alpha 12.0 \
    --input-norm Mean \
    --encoder-type None \
    --transform None \
    --extract-path Data/xvector/${model}${resnet_size}/${dataset}/${feat}/${loss} \
    --model ${model} \
    --resnet-size ${resnet_size} \
    --dropout-p 0.25 \
    --epoch 24
fi
