#!/usr/bin/env bash

stage=60

lstm_dir=/home/work2020/yangwenhao/project/lstm_speaker_verification


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
    --extract-path Data/xvector/${model}${resnet_size}/${dataset}/${feat}/${loss}_nan \
    --model ${model} \
    --resnet-size ${resnet_size} \
    --dropout-p 0.25 \
    --epoch 24
fi


if [ $stage -le 60 ]; then
  model=TDNN_v5
  dataset=vox1
  loss=soft
  feat_type=spect
  feat=log
  loss=arcsoft
  model=TDNN_v5
  encod=STAP
  dataset=vox1
  test_set=sitw

  for subset in test ; do # 32,128,512; 8,32,128
    echo -e "\n\033[1;4;31m Stage ${stage}: Testing ${model} in ${test_set} with ${loss} \033[0m\n"
    python -W ignore Xvector_Extraction/extract_xvector_egs.py \
      --model ${model} \
      --train-config-dir ${lstm_dir}/data/${dataset}/egs/${feat_type}/dev_${feat} \
      --train-dir ${lstm_dir}/data/${dataset}/${feat_type}/dev_${feat} \
      --test-dir ${lstm_dir}/data/${test_set}/${feat_type}/${subset}_${feat} \
      --feat-format kaldi \
      --input-norm Mean \
      --input-dim 161 \
      --nj 12 \
      --embedding-size 512 \
      --loss-type ${loss} \
      --encoder-type STAP \
      --channels 512,512,512,512,1500 \
      --margin 0.25 \
      --s 30 \
      --input-length var \
      --frame-shift 300 \
      --xvector-dir Data/xvector/TDNN_v5/aishell2/spect_egs_baseline/arcsoft_0ce/inputMean_STAP_em512_wde4/${test_set}_${subset}_var \
      --resume Data/checkpoint/TDNN_v5/vox2_v2/spect_egs/arcsoft_0ce/inputMean_STAP_em512_wde4/checkpoint_60.pth \
      --gpu-id 0 \
      --cos-sim
  done
  exit
fi