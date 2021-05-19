#!/usr/bin/env bash

stage=0
waited=0
while [ $(ps 103374 | wc -l) -eq 2 ]; do
  sleep 60
  waited=$(expr $waited + 1)
  echo -en "\033[1;4;31m Having waited for ${waited} minutes!\033[0m\r"
done
lstm_dir=/home/work2020/yangwenhao/project/lstm_speaker_verification

if [ $stage -le 0 ]; then
  datasets=vox1
  model=TDNN_v5
  feat_type=spect
  feat=log
  block_type=None
  input_norm=Mean
  dropout_p=0
  encoder_type=STAP
  #  loss=arcsoft
  loss=soft
  avgsize=4
  alpha=0
  embedding_size=256
  block_type=None
  filter=fBPLayer
  feat_dim=24
  lr_ratio=0.001

  for filter in fBPLayer fLlayer; do
    echo -e "\n\033[1;4;31m Stage${stage} :Training ${model} in vox1 with ${loss} kernel 5,5 \033[0m\n"
    python TrainAndTest/Spectrogram/train_egs.py \
      --model ${model} \
      --train-dir ${lstm_dir}/data/vox1/egs/spect/dev_${feat} \
      --train-test-dir ${lstm_dir}/data/vox1/spect/dev_${feat}/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/vox1/egs/spect/valid_${feat} \
      --test-dir ${lstm_dir}/data/vox1/spect/test_${feat} \
      --batch-size 128 \
      --input-norm ${input_norm} \
      --test-input fix \
      --feat-format kaldi \
      --nj 10 \
      --epochs 40 \
      --lr 0.1 \
      --input-dim 161 \
      --filter ${filter} \
      --time-dim 1 \
      --exp \
      --feat-dim ${feat_dim} \
      --scheduler rop \
      --patience 3 \
      --milestones 10,20,30 \
      --check-path Data/checkpoint/${model}/${datasets}/${feat_type}_egs_filter/${loss}_0ce/Input${input_norm}_${encoder_type}_${block_type}_dp${dropout_p}_avg${avgsize}_alpha${alpha}_em${embedding_size}_wd5e4/${filter}${feat_dim}_adalr${lr_ratio}_full \
      --resume Data/checkpoint/${model}/${datasets}/${feat_type}_egs_filter/${loss}_0ce/Input${input_norm}_${encoder_type}_${block_type}_dp${dropout_p}_avg${avgsize}_alpha${alpha}_em${embedding_size}_wd5e4/${filter}${feat_dim}_adalr${lr_ratio}_full/checkpoint_9.pth \
      --stride 1 \
      --block-type ${block_type} \
      --channels 512,512,512,512,1500 \
      --encoder-type ${encoder_type} \
      --embedding-size ${embedding_size} \
      --avg-size ${avgsize} \
      --alpha ${alpha} \
      --num-valid 2 \
      --margin 0.25 \
      --s 30 \
      --m 3 \
      --lr-ratio ${lr_ratio} \
      --weight-decay 0.0005 \
      --dropout-p ${dropout_p} \
      --gpu-id 0,1 \
      --cos-sim \
      --extract \
      --all-iteraion 0 \
      --loss-type ${loss}
  done
  exit
fi
