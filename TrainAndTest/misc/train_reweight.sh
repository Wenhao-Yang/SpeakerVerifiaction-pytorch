#!/usr/bin/env bash

# author: yangwenhao
# contact: 874681044@qq.com
# file: train_reweight.sh
# time: 2022/4/16 15:50
# Description: 

waited=0
while [ `ps 1141965 | wc -l` -eq 2 ]; do
  sleep 60
  waited=$(expr $waited + 1)
  echo -en "\033[1;4;31m Having waited for ${waited} minutes!\033[0m\r"
done
stage=1

lstm_dir=/home/work2020/yangwenhao/project/lstm_speaker_verification


if [ $stage -le 0 ]; then
  datasets=cnceleb
  testset=cnceleb
  feat_type=klfb
  model=ThinResNet
  resnet_size=18
  encoder_type=SAP2
  embedding_size=512
  block_type=basic
  downsample=k3
  kernel=5,5
  loss=arcdist

  alpha=0
  input_norm=Mean
#  mask_layer=None
  scheduler=rop
  optimizer=sgd
  input_dim=40
  batch_size=128 #384
  fast=none1
  mask_layer=baseline
  weight=vox2_rcf
  scale=0.2
  subset=
  stat_type=maxmargin
  loss_ratio=5
  chn=16
        # --milestones 15,25,35,45 \

  for loss in angleproto ; do
    echo -e "\n\033[1;4;31m Stage${stage}: Training ${model}${resnet_size} in ${datasets}_egs with ${loss} with ${input_norm} normalization \033[0m\n"

    if [ "$loss" == "arcdist" ];then
      loss_str=_${stat_type}lr${loss_ratio}
    else
      loss_str=
    fi

    if [ $chn -eq 16 ]; then
      channels=16,32,64,128
      chn_str=
    elif [ $chn -eq 32 ]; then
      channels=32,64,128,256
      chn_str=chn32_
    fi

#    ${loss_str}
    python TrainAndTest/misc/train_egs_reweight.py \
      --model ${model} \
      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev_fb${input_dim} \
      --train-test-dir ${lstm_dir}/data/${datasets}/${feat_type}/dev_fb${input_dim}/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev_fb${input_dim}_valid \
      --test-dir ${lstm_dir}/data/${testset}/${feat_type}/test_fb${input_dim} \
      --feat-format kaldi \
      --random-chunk 200 400 \
      --input-norm ${input_norm} \
      --resnet-size ${resnet_size} \
      --nj 0 \
      --epochs 50 \
      --batch-size ${batch_size} \
      --optimizer ${optimizer} \
      --scheduler ${scheduler} \
      --lr 0.1 \
      --base-lr 0.00000001 \
      --mask-layer ${mask_layer} \
      --init-weight ${weight} \
      --scale ${scale} \
      --milestones 10,20,30,40,50 \
      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs${subset}_${mask_layer}_reweight/${loss}_${optimizer}_${scheduler}/${chn_str}${input_norm}_batch${batch_size}_${block_type}_down${downsample}_${fast}_${encoder_type}_dp01_alpha${alpha}_em${embedding_size}${loss_str}_wde4_var \
      --resume Data/checkpoint/ThinResNet34/cnceleb/klfb_egs_baseline/arcsoft_sgd_rop/Mean_batch256_basic_downk3_none1_SAP2_dp01_alpha0_em512_wd5e4_var/checkpoint_60.pth \
      --kernel-size ${kernel} \
      --downsample ${downsample} \
      --channels ${channels} \
      --fast ${fast} \
      --stride 2,1 \
      --block-type ${block_type} \
      --embedding-size ${embedding_size} \
      --time-dim 1 \
      --avg-size 5 \
      --enroll-utts 2 \
      --num-meta-spks 40 \
      --encoder-type ${encoder_type} \
      --num-valid 2 \
      --alpha ${alpha} \
      --margin 0.2 \
      --m 0.2 \
      --s 30 \
      --num-center 3 \
      --weight-decay 0.0001 \
      --dropout-p 0.1 \
      --gpu-id 0 \
      --extract \
      --cos-sim \
      --all-iteraion 0 \
      --remove-vad \
      --loss-type ${loss} \
      --stat-type ${stat_type} \
      --loss-ratio ${loss_ratio}
  done
  # --lncl
  exit
fi

if [ $stage -le 1 ]; then
  datasets=cnceleb
  testset=cnceleb
  feat_type=klfb
  model=ThinResNet
  resnet_size=18
  encoder_type=SAP2
  embedding_size=512
  block_type=basic
  downsample=k3
  kernel=5,5
  loss=arcdist

  alpha=0
  input_norm=Mean
#  mask_layer=None
  scheduler=rop
  optimizer=sgd
  input_dim=40
  batch_size=192 #384
  fast=none1
  mask_layer=baseline
  weight=vox2_rcf
  scale=0.2
  subset=12
  stat_type=maxmargin
  loss_ratio=5
  chn=16
  train_subset=12
        # --milestones 15,25,35,45 \
  e2e_loss=angleproto

  for loss in arcsoft ; do
    echo -e "\n\033[1;4;31m Stage${stage}: Training ${model}${resnet_size} in ${datasets}_egs with ${loss} with ${input_norm} normalization \033[0m\n"

    if [ "$loss" == "arcdist" ];then
      loss_str=_${stat_type}lr${loss_ratio}
    else
      loss_str=
    fi

    if [ $chn -eq 16 ]; then
      channels=16,32,64,128
      chn_str=
    elif [ $chn -eq 32 ]; then
      channels=32,64,128,256
      chn_str=chn32_
    fi

#    ${loss_str}
    python TrainAndTest/misc/train_egs_reweight.py \
      --model ${model} \
      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev${train_subset}_fb${input_dim} \
      --train-test-dir ${lstm_dir}/data/${datasets}/${feat_type}/dev_fb${input_dim}/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev${train_subset}_fb${input_dim}_valid \
      --test-dir ${lstm_dir}/data/${testset}/${feat_type}/test_fb${input_dim} \
      --feat-format kaldi \
      --random-chunk 200 400 \
      --input-norm ${input_norm} \
      --resnet-size ${resnet_size} \
      --nj 0 \
      --epochs 50 \
      --batch-size ${batch_size} \
      --optimizer ${optimizer} \
      --scheduler ${scheduler} \
      --lr 0.1 \
      --base-lr 0.00000001 \
      --mask-layer ${mask_layer} \
      --init-weight ${weight} \
      --scale ${scale} \
      --milestones 10,20,30,40,50 \
      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs${subset}_${mask_layer}_reweight/${e2e_loss}${loss}_${optimizer}_${scheduler}/${chn_str}${input_norm}_batch${batch_size}_${block_type}_down${downsample}_${fast}_${encoder_type}_dp01_alpha${alpha}_em${embedding_size}${loss_str}_wde4_var \
      --kernel-size ${kernel} \
      --downsample ${downsample} \
      --channels ${channels} \
      --fast ${fast} \
      --stride 2,1 \
      --block-type ${block_type} \
      --embedding-size ${embedding_size} \
      --time-dim 1 \
      --avg-size 5 \
      --enroll-utts 2 \
      --num-meta-spks 40 \
      --encoder-type ${encoder_type} \
      --num-valid 2 \
      --alpha ${alpha} \
      --margin 0.2 \
      --m 0.2 \
      --s 30 \
      --num-center 3 \
      --weight-decay 0.0001 \
      --dropout-p 0.1 \
      --gpu-id 0 \
      --extract \
      --cos-sim \
      --all-iteraion 0 \
      --remove-vad \
      --loss-type ${loss} \
      --e2e-loss-type ${e2e_loss} \
      --stat-type ${stat_type} \
      --loss-ratio ${loss_ratio}
  done
#        --resume Data/checkpoint/ThinResNet34/cnceleb/klfb_egs_baseline/arcsoft_sgd_rop/Mean_batch256_basic_downk3_none1_SAP2_dp01_alpha0_em512_wde4_var/checkpoint_60.pth \

  # --lncl
  exit
fi