#!/usr/bin/env bash

stage=50

waited=0
while [ $(ps 113458 | wc -l) -eq 2 ]; do
  sleep 60
  waited=$(expr $waited + 1)
  echo -en "\033[1;4;31m Having waited for ${waited} minutes!\033[0m\r"
done
lstm_dir=/home/work2020/yangwenhao/project/lstm_speaker_verification

#stage=1

if [ $stage -le 30 ]; then
  datasets=cnceleb
  model=DomResNet
  resnet_size=8
  kernel_size=5,5
  channels=16,32,128
  for loss in soft; do
    echo -e "\033[1;4;31m Train ${model} with ${loss} loss in ${datasets}, \n    kernel_size is ${kernel_size} for connection, channels are ${channels}.\033[0m\n"
    python TrainAndTest/Spectrogram/train_domres_kaldi.py \
      --model ${model} \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/cnceleb/spect/dev_04 \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/cnceleb/spect/test \
      --feat-format npy \
      --resnet-size ${resnet_size} \
      --nj 10 \
      --epochs 15 \
      --lr 0.1 \
      --milestones 7,11 \
      --kernel-size ${kernel_size} \
      --check-path Data/checkpoint/${model}_lstm/${datasets}/spect_04/${loss}_sim \
      --resume Data/checkpoint/${model}_lstm/${datasets}/spect_04/${loss}_sim/checkpoint_1.pth \
      --channels ${channels} \
      --inst-norm \
      --embedding-size-a 128 \
      --embedding-size-b 128 \
      --embedding-size-o 0 \
      --input-per-spks 192 \
      --num-valid 1 \
      --domain \
      --alpha 9 \
      --dom-ratio 0.1 \
      --loss-ratio 0.05 \
      --weight-decay 0.001 \
      --dropout-p 0.25 \
      --gpu-id 0 \
      --loss-type ${loss}
  done
fi

#stage=100

if [ $stage -le 40 ]; then
  datasets=cnceleb
  model=DomResNet
  resnet_size=8
  kernel_size=5,5
  channels=16,64,128
  for loss in soft; do
    echo -e "\033[1;4;31m Train ${model} with ${loss} loss in ${datasets}, \n    kernel_size is ${kernel_size} for connection, channels are ${channels}.\033[0m\n"
    python TrainAndTest/Spectrogram/train_domres_kaldi.py \
      --model ${model} \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/cnceleb/spect/dev_04 \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/cnceleb/spect/test \
      --feat-format npy \
      --resnet-size ${resnet_size} \
      --nj 10 \
      --epochs 15 \
      --lr 0.1 \
      --milestones 7,11 \
      --kernel-size ${kernel_size} \
      --check-path Data/checkpoint/${model}/${datasets}/spect_04/${loss}_sim \
      --resume Data/checkpoint/${model}/${datasets}/spect_04/${loss}_sim/checkpoint_1.pth \
      --channels ${channels} \
      --embedding-size-a 128 \
      --embedding-size-b 128 \
      --embedding-size-o 0 \
      --input-per-spks 192 \
      --num-valid 1 \
      --domain \
      --alpha 9 \
      --dom-ratio 0.1 \
      --loss-ratio 0.05 \
      --weight-decay 0.001 \
      --dropout-p 0.25 \
      --gpu-id 0 \
      --loss-type ${loss}
  done
fi

if [ $stage -le 50 ]; then
  datasets=cnceleb
  model=TDNN_v5
  feat_type=pyfb
  loss=soft
  encod=STAP
  embedding_size=256
  input_dim=40
  input_norm=Mean
  lr_ratio=1
  loss_ratio=1
  feat=fb${input_dim}_ws25
  #  resnet_size=8
  #  kernel_size=5,5
  #  channels=
  for loss in soft; do
    echo -e "\033[1;4;31m Stage ${stage}: Train ${model} with ${loss} loss in ${datasets}.\033[0m\n"
    python TrainAndTest/train_egs_domain.py \
      --model ${model} \
      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev_${feat} \
      --train-test-dir ${lstm_dir}/data/${datasets}/${feat_type}/dev_${feat}/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/valid_${feat} \
      --test-dir ${lstm_dir}/data/${datasets}/${feat_type}/test_${feat} \
      --remove-vad \
      --feat-format kaldi \
      --shuffle \
      --random-chunk 200 400 \
      --nj 10 \
      --epochs 60 \
      --lr 0.1 \
      --milestones 12,24,36,48 \
      --input-dim ${input_dim} \
      --channels 512,512,512,512,1500 \
      --encoder-type ${encod} \
      --check-path Data/checkpoint/${model}/${datasets}/${feat_type}_egs_revg/${loss}/feat${feat}_input${input_norm}_${encod}_em${embedding_size}_wde3_step5_domain2dr1 \
      --resume Data/checkpoint/${model}/${datasets}/${feat_type}_egs_revg/${loss}/feat${feat}_input${input_norm}_${encod}_em${embedding_size}_wde3_step5_domain2dr1/checkpoint_21.pth \
      --embedding-size ${embedding_size} \
      --stride 1 \
      --num-valid 1 \
      --domain \
      --domain-steps 5 \
      --dom-ratio 1 \
      --loss-ratio 0.05 \
      --sim-ratio 0 \
      --weight-decay 0.001 \
      --dropout-p 0 \
      --gpu-id 0,1 \
      --loss-type ${loss}
  done
fi
