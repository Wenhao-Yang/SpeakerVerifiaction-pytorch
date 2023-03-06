#!/usr/bin/env bash

stage=0
if [ $stage -le 0 ]; then
  lstm_dir=/home/work2020/yangwenhao/project/lstm_speaker_verification
  datasets=army
  model=MultiResNet
  resnet_size=10
  loss=soft
  encod=None
  transform=None
  loss_ratio=0.01
  for loss in center ; do
    echo -e "\n\033[1;4;31m Training ${model} in vox1 with ${loss} kernel 5,5 \033[0m\n"

    python TrainAndTest/Spectrogram/train_egs_multi.py \
      --model ${model} \
      --train-dir-a ${lstm_dir}/data/${datasets}/egs/spect/aishell2_dev_8k_v2 \
      --train-dir-b ${lstm_dir}/data/${datasets}/egs/spect/vox1_dev_8k_v2 \
      --train-test-dir ${lstm_dir}/data/${datasets}/spect/dev_8k_v2 \
      --valid-dir-a ${lstm_dir}/data/${datasets}/egs/spect/aishell2_valid_8k_v2 \
      --valid-dir-b ${lstm_dir}/data/${datasets}/egs/spect/vox1_valid_8k_v2 \
      --test-dir ${lstm_dir}/data/${datasets}/spect/test_8k \
      --feat-format kaldi \
      --resnet-size ${resnet_size} \
      --input-norm Mean \
      --batch-size 192 \
      --nj 12 \
      --epochs 1 \
      --lr 0.001 \
      --input-dim 81 \
      --stride 1 \
      --milestones 3 \
      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}_x2/spect_egs_${encod}/${loss}_dp25_b192_16_${loss_ratio} \
      --resume Data/checkpoint/${model}${resnet_size}/${datasets}_x2/spect_egs_${encod}/${loss}_dp25_b192_16_${loss_ratio}/checkpoint_24.pth \
      --channels 16,64,128,256 \
      --embedding-size 128 \
      --transform ${transform} \
      --encoder-type ${encod} \
      --time-dim 1 \
      --avg-size 4 \
      --num-valid 4 \
      --alpha 12 \
      --margin 0.3 \
      --s 30 \
      --m 3 \
      --loss-ratio ${loss_ratio} \
      --set-ratio 0.6 \
      --grad-clip 0 \
      --weight-decay 0.001 \
      --dropout-p 0.25 \
      --gpu-id 0 \
      --cos-sim \
      --extract \
      --loss-type ${loss}
  done
fi