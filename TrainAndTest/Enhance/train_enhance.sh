#!/usr/bin/env bash

stage=1
lstm_dir=/home/work2020/yangwenhao/project/lstm_speaker_verification

# ===============================    LoResNet10    ===============================
if [ $stage -le 1 ]; then
  datasets=vox1
  model=Demucs
  resnet_size=4

  loss=mse
  feat_type=klfb
  sname=dev_auged2_fb40_pair
  downsample=k3
  input_norm=none
  #        --scheduler cyclic \
#  for block_type in seblock cbam; do
  for downsample in k5; do
    echo -e "\n\033[1;4;31mStage ${stage}: Training ${model}${resnet_size} in ${datasets}_egs with ${loss} \033[0m\n"
    python TrainAndTest/train_egs_enhance.py \
      --model ${model} \
      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname} \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname}_valid \
      --feat-format kaldi \
      --input-norm ${input_norm} \
      --random-chunk 150 400 \
      --resnet-size ${resnet_size} \
      --nj 12 \
      --epochs 210 \
      --optimizer adam \
      --scheduler cyclic \
      --patience 3 \
      --accu-steps 1 \
      --stride 2,2,2,2 \
      --lr 0.001 \
      --milestones 10,20,30,40 \
      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs2_enhance/${loss}/wde5_cyclic_cliplen_var275 \
      --resume Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs2_enhance/${loss}/wde5_cyclic_cliplen_var275/checkpoint_40.pth \
      --kernel-size 3 \
      --shuffle \
      --input-dim 40 \
      --batch-size 128 \
      --num-valid 2 \
      --grad-clip 0 \
      --s 30 \
      --lr-ratio 0.01 \
      --weight-decay 0.00001 \
      --gpu-id 0,1 \
      --all-iteraion 0 \
      --cos-sim \
      --loss-type ${loss}
  done
fi