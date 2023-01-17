#!/usr/bin/env bash

stage=80

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
    --dropout-p 0.0 \
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

if [ $stage -le 3 ]; then
  model=LoResNet
  dataset=vox1
  loss=soft
  feat_type=spect
  feat=log
  loss=soft
  encod=None
  test_set=vox1
  resnet_size=8
  block_type=cbam

  for subset in test; do # 32,128,512; 8,32,128
    echo -e "\n\033[1;4;31m Stage ${stage}: Testing ${model} in ${test_set} with ${loss} \033[0m\n"
    python -W ignore Extraction/extract_xvector_egs.py \
      --model ${model} \
      --train-config-dir ${lstm_dir}/data/${dataset}/egs/${feat_type}/dev_${feat} \
      --train-dir ${lstm_dir}/data/${dataset}/${feat_type}/dev_log \
      --test-dir ${lstm_dir}/data/${test_set}/${feat_type}/${subset}_${feat} \
      --feat-format kaldi \
      --input-norm Mean \
      --input-dim 40 \
      --nj 12 \
      --resnet-size ${resnet_size} \
      --embedding-size 256 \
      --loss-type ${loss} \
      --encoder-type None \
      --avg-size 4 \
      --time-dim 1 \
      --stride 1 \
      --block-type ${block_type} \
      --channels 64,128,256 \
      --margin 0.25 \
      --s 30 \
      --xvector \
      --frame-shift 300 \
      --xvector-dir Data/xvector/LoResNet8/vox1/spect_egs/soft/None_cbam_dp25_alpha12_em256/${test_set}_${subset}_var \
      --resume Data/checkpoint/LoResNet8/vox1/spect_egs/soft/None_cbam_dp25_alpha12_em256/checkpoint_5.pth \
      --gpu-id 0 \
      --remove-vad \
      --cos-sim
  done
  exit
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
  dataset=vox2
  loss=soft
  feat_type=spect
  feat=log
  loss=arcsoft
  model=TDNN_v5
  encod=STAP
  dataset=vox2
  test_set=cnceleb
  #       --train-dir ${lstm_dir}/data/${dataset}/${feat_type}/dev_${feat} \

  for subset in test; do # 32,128,512; 8,32,128
    echo -e "\n\033[1;4;31m Stage ${stage}: Testing ${model} in ${test_set} with ${loss} \033[0m\n"
    python -W ignore Extraction/extract_xvector_egs.py \
      --model ${model} \
      --train-config-dir ${lstm_dir}/data/${dataset}/egs/${feat_type}/dev_${feat} \
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
      --frame-shift 300 \
      --xvector-dir Data/xvector/TDNN_v5/vox2_v2/spect_egs/arcsoft_0ce/inputMean_STAP_em512_wde4/${test_set}_${subset}_var \
      --resume Data/checkpoint/TDNN_v5/vox2_v2/spect_egs/arcsoft_0ce/inputMean_STAP_em512_wde4/checkpoint_60.pth \
      --gpu-id 0 \
      --cos-sim
  done
  exit
fi

if [ $stage -le 70 ]; then
  model=TDNN_v5
  dataset=vox1
  loss=soft
  feat_type=pyfb
  feat=fb40
  loss=soft
  model=TDNN_v5
  encod=STAP
  test_set=vox1

  for subset in test; do # 32,128,512; 8,32,128
    echo -e "\n\033[1;4;31m Stage ${stage}: Testing ${model} in ${test_set} with ${loss} \033[0m\n"
    python -W ignore Extraction/extract_xvector_egs.py \
      --model ${model} \
      --train-config-dir ${lstm_dir}/data/${dataset}/egs/${feat_type}/dev_${feat}_ws25 \
      --train-dir ${lstm_dir}/data/${dataset}/${feat_type}/dev_${feat}_ws25 \
      --test-dir ${lstm_dir}/data/${test_set}/${feat_type}/${subset}_${feat}_ws25 \
      --feat-format kaldi \
      --input-norm Mean \
      --input-dim 40 \
      --nj 12 \
      --embedding-size 256 \
      --loss-type ${loss} \
      --encoder-type STAP \
      --channels 512,512,512,512,1500 \
      --margin 0.25 \
      --s 30 \
      --xvector \
      --frame-shift 300 \
      --xvector-dir Data/xvector/TDNN_v5/vox1/pyfb_egs_baseline/soft/featfb40_ws25_inputMean_STAP_em256_wde3_var/${test_set}_${subset}_var \
      --resume Data/checkpoint/TDNN_v5/vox1/pyfb_egs_baseline/soft/featfb40_ws25_inputMean_STAP_em256_wde3_var/checkpoint_50.pth \
      --gpu-id 0 \
      --remove-vad \
      --cos-sim
  done
  exit
fi

if [ $stage -le 71 ]; then
  model=TDNN_v5
  dataset=vox1
  loss=soft
  feat_type=klfb
  feat=combined
  loss=soft
  model=TDNN_v5
  encod=STAP
  test_set=vox1

  for subset in test; do # 32,128,512; 8,32,128
    echo -e "\n\033[1;4;31m Stage ${stage}: Testing ${model} in ${test_set} with ${loss} \033[0m\n"
    python -W ignore Extraction/extract_xvector_egs.py \
      --model ${model} \
      --train-config-dir ${lstm_dir}/data/${dataset}/egs/${feat_type}/dev_${feat}_fb40 \
      --train-dir ${lstm_dir}/data/${dataset}/${feat_type}/dev_combined \
      --test-dir ${lstm_dir}/data/${test_set}/${feat_type}/${subset}_fb40 \
      --feat-format kaldi \
      --input-norm Mean \
      --input-dim 40 \
      --nj 12 \
      --embedding-size 512 \
      --loss-type ${loss} \
      --encoder-type STAP \
      --channels 512,512,512,512,1500 \
      --margin 0.25 \
      --s 30 \
      --xvector \
      --frame-shift 300 \
      --xvector-dir Data/xvector/TDNN_v5/vox1/klfb_egs_baseline/soft/featcombined_inputMean_STAP_em512_wde3_var_v3/${test_set}_${subset}_var \
      --resume Data/checkpoint/TDNN_v5/vox1/klfb_egs_baseline/soft/featcombined_inputMean_STAP_em512_wde3_var_v2/checkpoint_50.pth \
      --gpu-id 0 \
      --remove-vad \
      --cos-sim
  done
  exit
fi

if [ $stage -le 80 ]; then
  model=TDNN_v5
  dataset=vox1
  loss=soft
  feat_type=pyfb
  feat=fb40
  loss=soft
  model=TDNN_v5
  encod=STAP
  dataset=vox2
  test_set=vox1

  for subset in test; do # 32,128,512; 8,32,128
    echo -e "\n\033[1;4;31m Stage ${stage}: Testing ${model} in ${test_set} with ${loss} \033[0m\n"
    python -W ignore Extraction/extract_xvector_egs.py \
      --model ${model} \
      --train-config-dir ${lstm_dir}/data/${dataset}/egs/${feat_type}/dev_${feat}_ws25 \
      --train-dir ${lstm_dir}/data/${dataset}/${feat_type}/dev_${feat} \
      --test-dir ${lstm_dir}/data/${test_set}/${feat_type}/${subset}_${feat}_ws25 \
      --feat-format kaldi \
      --input-norm Mean \
      --input-dim 40 \
      --nj 12 \
      --embedding-size 512 \
      --loss-type ${loss} \
      --encoder-type STAP \
      --channels 512,512,512,512,1500 \
      --margin 0.25 \
      --s 30 \
      --frame-shift 300 \
      --xvector \
      --xvector-dir Data/xvector/TDNN_v5/vox2/pyfb_egs_baseline/soft/featfb40_ws25_inputMean_STAP_em512_wd5e4_var/${test_set}_${subset}_var \
      --resume Data/checkpoint/TDNN_v5/vox2/pyfb_egs_baseline/soft/featfb40_ws25_inputMean_STAP_em512_wd5e4_var/checkpoint_40.pth \
      --gpu-id 0 \
      --remove-vad \
      --cos-sim
  done
  exit
fi


if [ $stage -le 100 ]; then
  model=ThinResNet resnet_size=34
  input_dim=40 feat_type=klfb
  feat=fb${input_dim}
  input_norm=Mean
  loss=arcsoft

#  encoder_type=SAP2 embedding_size=512
  # encoder_type=SAP2 embedding_size=256
  encoder_type=ASTP2 embedding_size=256
  # block_type=seblock downsample=k1 red_ratio=2
  # block_type=cbam downsample=k3 red_ratio=2
  kernel=5,5 fast=none1
  loss=arcsoft
  alpha=0

  batch_size=256
  mask_layer=baseline mask_len=5,5
  # train_set=cnceleb test_set=cnceleb
  train_set=vox1 test_set=vox1
  train_subset=
#  subset=dev
  subset=dev test_input=fix
  epoch=13

#     --trials subtrials/trials_${s} --score-suffix ${s} \
# Data/checkpoint/ThinResNet34/cnceleb/klfb80_egs_baseline/arcsoft_sgd_rop/Mean_batch256_basic_downk3_none1_SAP2_dp01_alpha0_em512_wd5e4_var
#--score-norm as-norm --n-train-snts 100000 --cohort-size 5000 \
#     --vad-select \

echo -e "\n\033[1;4;31m Stage${stage}: Test ${model}${resnet_size} in ${test_set}_egs with ${loss} with ${input_norm} normalization \033[0m\n"

for seed in 123456 ; do
for train_set in cnceleb vox1 ; do
  s=all
  for ((epoch=0; epoch<=49; i=i+1)); do
  # for epoch in 13 ; do     #1 2 5 6 9 10 12 13 17 20 21 25 26 27 29 30 33 37 40 4

    model_dir=${model}${resnet_size}/${train_set}/klfb_egs_baseline/arcsoft_sgd_rop/Mean_batch256_seblock_red2_downk1_avg5_ASTP2_em256_dp01_alpha0_none1_wde4_varesmix2_bashuf2_dist_core/percent0.5_random/123456
    
   python -W ignore TrainAndTest/test_egs.py \
     --model ${model} --resnet-size ${resnet_size} \
     --train-dir ${lstm_dir}/data/${train_set}/${feat_type}/dev${train_subset}_${feat} \
     --train-extract-dir ${lstm_dir}/data/${train_set}/${feat_type}/dev${train_subset}_${feat} \
     --train-test-dir ${lstm_dir}/data/${train_set}/${feat_type}/dev_${feat}/trials_dir \
     --train-trials trials_2w \
     --valid-dir ${lstm_dir}/data/${train_set}/${feat_type}/dev${train_subset}_${feat}_valid \
     --test-dir ${lstm_dir}/data/${train_set}/${feat_type}/${subset}_${feat} \
     --feat-format kaldi --nj 6 --remove-vad \
     --input-norm ${input_norm} --input-dim ${input_dim} \
     --mask-layer ${mask_layer} --mask-len ${mask_len} \
     --kernel-size ${kernel} --fast ${fast} --stride 2,1 \
     --channels 16,32,64,128 \
     --time-dim 1 --avg-size 5 \
     --loss-type ${loss} --margin 0.15 --s 30 \
     --block-type ${block_type} --downsample ${downsample} --red-ratio ${red_ratio} \
     --encoder-type ${encoder_type} --embedding-size ${embedding_size} --alpha 0 \
     --test-input ${test_input} --frame-shift 300 \
     --xvector-dir Data/xvector/${model_dir}/${test_set}_${subset}_${epoch}_${test_input} \
     --resume Data/checkpoint/${model_dir}/checkpoint_${epoch}.pth \
     --gpu-id 2 --verbose 0 \
     --cos-sim
     # checkpoint_${epoch}.pth _${epoch}
#     --extract \
#      --model-yaml Data/checkpoint/ThinResNet34/cnceleb/klfb40_egs12_baseline/arcsoft_sgd_rop/Mean_batch256_basic_downk3_none1_SAP2_dp01_alpha0_em512_wd5e4_var/model.2022.02.22.yaml \
 done
done
done


fi