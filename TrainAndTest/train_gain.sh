#!/usr/bin/env bash

stage=62
lstm_dir=/home/work2020/yangwenhao/project/lstm_speaker_verification
waited=0
while [ $(ps 113458 | wc -l) -eq 2 ]; do
  sleep 60
  waited=$(expr $waited + 1)
  echo -en "\033[1;4;31m Having waited for ${waited} minutes!\033[0m\r"
done

if [ $stage -le 62 ]; then
  datasets=vox1
  model=ThinResNet
  resnet_size=34
  feat_type=spect
  feat=log
  block_type=None
  dropout_p=0
  encoder_type=STAP
  loss=soft
  avgsize=4
  alpha=0
  embedding_size=128
  feat_dim=24

  for block_type in None; do
    echo -e "\n\033[1;4;31m Training ${model} in vox1 with ${loss} kernel 5,5 \033[0m\n"
    python TrainAndTest/train_gain.py \
      --model ${model} \
      --train-dir ${lstm_dir}/data/vox1/egs/spect/dev_${feat} \
      --train-test-dir ${lstm_dir}/data/vox1/spect/dev_${feat}/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/vox1/egs/spect/valid_${feat} \
      --test-dir ${lstm_dir}/data/vox1/spect/test_${feat} \
      --batch-size 128 \
      --input-norm Mean \
      --exp \
      --test-input fix \
      --feat-format kaldi \
      --resnet-size ${resnet_size} \
      --nj 10 \
      --epochs 50 \
      --lr 0.1 \
      --input-dim 161 \
      --feat-dim ${feat_dim} \
      --scheduler rop \
      --milestones 10,20,25 \
      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_mean/${loss}/gain_sig_f/0.1_clamp_${encoder_type}_${block_type}_dp${dropout_p}_avg${avgsize}_alpha${alpha}_em${embedding_size}_wde4 \
      --resume Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_mean/${loss}/gain_sig_f/0.1_clamp_${encoder_type}_${block_type}_dp${dropout_p}_avg${avgsize}_alpha${alpha}_em${embedding_size}_wde4/checkpoint_9.pth \
      --stride 2 \
      --block-type ${block_type} \
      --channels 16,32,64,128 \
      --encoder-type ${encoder_type} \
      --embedding-size ${embedding_size} \
      --time-dim 1 \
      --avg-size ${avgsize} \
      --alpha ${alpha} \
      --num-valid 2 \
      --margin 0.2 \
      --s 30 \
      --m 3 \
      --loss-ratio 0.1 \
      --weight-decay 0.0001 \
      --dropout-p ${dropout_p} \
      --gpu-id 2 \
      --cos-sim \
      --extract \
      --all-iteraion 0 \
      --loss-type ${loss}
  done
  exit
fi

if [ $stage -le 72 ]; then
  datasets=vox1
  model=LoResNet
  resnet_size=8
  feat_type=spect
  feat=log
  block_type=cbam
  dropout_p=0.25
  encoder_type=None
  loss=soft
  avgsize=4
  alpha=12
  embedding_size=128
  #  feat_dim=24
  #--feat-dim ${feat_dim} \

  for block_type in None; do
    echo -e "\n\033[1;4;31m Training ${model} in vox1 with ${loss} kernel 5,5 \033[0m\n"
    python TrainAndTest/train_gain.py \
      --model ${model} \
      --train-dir ${lstm_dir}/data/vox1/egs/spect/dev_${feat} \
      --train-test-dir ${lstm_dir}/data/vox1/spect/dev_${feat}/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/vox1/egs/spect/valid_${feat} \
      --test-dir ${lstm_dir}/data/vox1/spect/test_${feat} \
      --batch-size 128 \
      --input-norm Mean \
      --exp \
      --test-input fix \
      --feat-format kaldi \
      --resnet-size ${resnet_size} \
      --nj 10 \
      --epochs 40 \
      --lr 0.1 \
      --input-dim 161 \
      --scheduler rop \
      --milestones 10,20,30 \
      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_mean/${loss}/gain_sig_f/0.1_clamp_${encoder_type}_${block_type}_dp${dropout_p}_avg${avgsize}_alpha${alpha}_em${embedding_size}_wde4 \
      --resume Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_mean/${loss}/gain_sig_f/0.1_clamp_${encoder_type}_${block_type}_dp${dropout_p}_avg${avgsize}_alpha${alpha}_em${embedding_size}_wde4/checkpoint_9.pth \
      --stride 2 \
      --block-type ${block_type} \
      --channels 64,128,256 \
      --encoder-type ${encoder_type} \
      --embedding-size ${embedding_size} \
      --time-dim 1 \
      --avg-size ${avgsize} \
      --alpha ${alpha} \
      --num-valid 2 \
      --margin 0.2 \
      --s 30 \
      --m 3 \
      --loss-ratio 0.1 \
      --weight-decay 0.001 \
      --dropout-p ${dropout_p} \
      --gpu-id 2 \
      --cos-sim \
      --extract \
      --all-iteraion 0 \
      --loss-type ${loss}
  done
fi
