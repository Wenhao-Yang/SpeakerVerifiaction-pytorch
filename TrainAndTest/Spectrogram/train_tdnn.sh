#!/usr/bin/env bash

# author: yangwenhao
# contact: 874681044@qq.com
# file: train_tdnn.sh
# time: 2022/5/21 08:55
# Description: 

stage=0

waited=0
while [ $(ps 17809 | wc -l) -eq 2 ]; do
  sleep 60
  waited=$(expr $waited + 1)
  echo -en "\033[1;4;31m Having waited for ${waited} minutes!\033[0m\r"
done
lstm_dir=/home/yangwenhao/project/lstm_speaker_verification


if [ $stage -le 0 ]; then
  datasets=vox2
  testset=vox1
  model=ECAPA
  resnet_size=34
  encoder_type=ASTP
  alpha=0
  block_type=res2tdnn
  embedding_size=192
  input_norm=Mean
  loss=arcsoft
  feat_type=klsp
  sname=dev

  mask_layer=baseline
  scheduler=rop
  optimizer=sgd
  input_dim=161
  batch_size=128
  chn=512
#  fast=none1
#  downsample=k5
  for seed in 123456 123457 123458 ; do
  for sname in dev ; do
    if [ $chn -eq 512 ]; then
      channels=512,512,512,512,1536
      chn_str=
    elif [ $chn -eq 1024 ]; then
      channels=1024,1024,1024,1024,3072
      chn_str=chn1024_
    fi
    echo -e "\n\033[1;4;31mStage ${stage}: Training ${model} in ${datasets}_egs with ${loss} \033[0m\n"
    model_dir=${model}/${datasets}/${feat_type}_egs_${mask_layer}/${loss}_${optimizer}_${scheduler}/${input_norm}_batch${batch_size}_${block_type}_${encoder_type}_em${embedding_size}_${chn_str}wd2e5_vares_bashuf/${seed}
    python TrainAndTest/train_egs.py \
      --model ${model} \
      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname} \
      --train-test-dir ${lstm_dir}/data/${testset}/${feat_type}/test \
      --train-trials trials \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname}_valid \
      --test-dir ${lstm_dir}/data/${testset}/${feat_type}/test \
      --feat-format kaldi \
      --input-norm ${input_norm} \
      --input-dim ${input_dim} \
      --batch-size ${batch_size} \
      --resnet-size ${resnet_size} \
      --nj 6 \
      --epochs 80 \
      --random-chunk 200 400 \
      --optimizer ${optimizer} \
      --scheduler ${scheduler} \
      --patience 2 \
      --early-stopping \
      --early-patience 20 \
      --cyclic-epoch 4 \
      --early-delta 0.0001 \
      --early-meta EER \
      --accu-steps 1 \
      --lr 0.1 \
      --base-lr 0.000001 \
      --milestones 10,20,40,50 \
      --check-path Data/checkpoint/${model_dir} \
      --resume Data/checkpoint/${model_dir}/checkpoint_21.pth \
      --channels ${channels} \
      --embedding-size ${embedding_size} \
      --encoder-type ${encoder_type} \
      --alpha ${alpha} \
      --margin 0.2 \
      --grad-clip 0 \
      --s 30 \
      --lr-ratio 0.01 \
      --weight-decay 0.00002 \
      --gpu-id 2,3 \
      --shuffle \
      --batch-shuffle \
      --all-iteraion 0 \
      --extract \
      --cos-sim \
      --loss-type ${loss}
  done
  done
  exit
fi



if [ $stage -le 10 ]; then
  model=ECAPA
  datasets=vox2
  #  feat=fb24
#  feat_type=pyfb
  feat_type=klsp
  loss=arcsoft
  encod=SAP2
  embedding_size=512
  input_dim=40
  input_norm=Mean
  lr_ratio=0
  loss_ratio=10
  subset=
  activation=leakyrelu
  scheduler=cyclic
  optimizer=adam
  stat_type=margin1 #margin1sum
  m=1.0

  # _lrr${lr_ratio}_lsr${loss_ratio}

 for stat_type in margin1 ; do
   feat=fb${input_dim}

   echo -e "\n\033[1;4;31m Stage ${stage}: Training ${model}_${encod} in ${datasets}_${feat} with ${loss}\033[0m\n"
   CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 TrainAndTest/train_egs_distributed.py \
   --train-config TrainAndTest/Spectrogram/TDNNs/vox2_ecapa.yaml
  done
  exit
fi