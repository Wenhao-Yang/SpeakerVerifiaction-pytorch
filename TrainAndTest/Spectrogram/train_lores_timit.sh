#!/usr/bin/env bash

#stage=3
stage=60

waited=0
while [ `ps 27212 | wc -l` -eq 2 ]; do
  sleep 60
  waited=$(expr $waited + 1)
  echo -en "\033[1;4;31m Having waited for ${waited} minutes!\033[0m\r"
done

if [ $stage -le 0 ]; then
  datasets=timit
#  loss=soft
  for loss in soft ; do
    echo -e "\n\033[1;4;31m Training with ${loss}\033[0m\n"
    python TrainAndTest/Spectrogram/train_lores10_var.py \
      --model LoResNet \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/spect/train \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/spect/test \
      --nj 8 \
      --epochs 12 \
      --lr 0.1 \
      --milestones 6,10 \
      --check-path Data/checkpoint/LoResNet8/${datasets}/spect_max_75/${loss}_var \
      --resume Data/checkpoint/LoResNet8/${datasets}/spect_max_75/${loss}_var/checkpoint_7.pth \
      --channels 4,16,64 \
      --embedding-size 128 \
      --input-per-spks 256 \
      --num-valid 1 \
      --weight-decay 0.001 \
      --alpha 10.8 \
      --dropout-p 0.5 \
      --gpu-id 0 \
      --loss-type ${loss}
  done
fi

if [ $stage -le 1 ]; then
  datasets=timit
#  loss=soft
  for loss in soft ; do
    echo -e "\n\033[1;4;31m Training with ${loss} minus mean \033[0m\n"
    python TrainAndTest/Spectrogram/train_lores10_var.py \
      --model GradResNet \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/spect/train_power \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/spect/test_power \
      --nj 8 \
      --epochs 12 \
      --lr 0.1 \
      --milestones 6,10 \
      --check-path Data/checkpoint/GradResNet8/${datasets}/spect_power_inst/${loss}_var \
      --resume Data/checkpoint/GradResNet8/${datasets}/spect_power_inst/${loss}_var/checkpoint_12.pth \
      --channels 4,16,64 \
      --embedding-size 128 \
      --input-per-spks 256 \
      --num-valid 1 \
      --inst-norm \
      --weight-decay 0.001 \
      --alpha 10.8 \
      --dropout-p 0.5 \
      --gpu-id 0 \
      --loss-type ${loss}
  done
fi

stage=100
if [ $stage -le 2 ]; then
  datasets=vox1
#  loss=soft
  for loss in soft ; do
    echo -e "\n\033[1;4;31m Training with ${loss}\033[0m\n"
    python TrainAndTest/Spectrogram/train_lores10_var.py \
      --model LoResNet \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/vox1/spect/dev_3w_wcmvn \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/vox1/spect/test_3w_wcmvn \
      --nj 10 \
      --epochs 12 \
      --lr 0.1 \
      --milestones 6,10 \
      --check-path Data/checkpoint/LoResNet8/${datasets}_3w_wcmvn/spect/${loss}_var \
      --resume Data/checkpoint/LoResNet8/${datasets}_3w_wcmvn/spect/${loss}_var/checkpoint_7.pth \
      --channels 4,16,64 \
      --embedding-size 128 \
      --input-per-spks 256 \
      --num-valid 1 \
      --weight-decay 0.001 \
      --alpha 10 \
      --feat-format kaldi \
      --dropout-p 0.5 \
      --gpu-id 0 \
      --loss-type ${loss}
  done
fi
if [ $stage -le 3 ]; then
  datasets=vox1_3w_power_25
  model=GradResNet
#  loss=soft
  for loss in soft ; do
    echo -e "\n\033[1;4;31m Training with ${loss}\033[0m\n"
    python TrainAndTest/Spectrogram/train_lores10_var.py \
      --model ${model} \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/vox1/spect/dev_3w_power_25 \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/vox1/spect/test_3w_power_25 \
      --nj 10 \
      --epochs 12 \
      --lr 0.1 \
      --milestones 6,10 \
      --check-path Data/checkpoint/${model}8/${datasets}/spect/${loss}_var \
      --resume Data/checkpoint/${model}8/${datasets}/spect/${loss}_var/checkpoint_7.pth \
      --channels 4,16,64 \
      --embedding-size 128 \
      --input-per-spks 256 \
      --num-valid 1 \
      --weight-decay 0.001 \
      --alpha 10 \
      --feat-format kaldi \
      --dropout-p 0.5 \
      --gpu-id 0 \
      --loss-type ${loss}
  done
fi

if [ $stage -le 4 ]; then
  datasets=vox1_3w_power_25
  model=GradResNet
#  loss=soft
  for loss in soft ; do
    echo -e "\n\033[1;4;31m Training with ${loss}\033[0m\n"
    python TrainAndTest/Spectrogram/train_lores10_var.py \
      --model ${model} \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/vox1/spect/dev_3w_power_25 \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/vox1/spect/test_3w_power_25 \
      --nj 10 \
      --epochs 12 \
      --lr 0.1 \
      --milestones 6,10 \
      --check-path Data/checkpoint/${model}8_32/${datasets}/spect/${loss}_var \
      --resume Data/checkpoint/${model}8_32/${datasets}/spect/${loss}_var/checkpoint_7.pth \
      --channels 4,32,64 \
      --embedding-size 128 \
      --input-per-spks 256 \
      --num-valid 1 \
      --weight-decay 0.001 \
      --alpha 10 \
      --feat-format kaldi \
      --dropout-p 0.5 \
      --gpu-id 0 \
      --loss-type ${loss}
  done
fi

stage=60
if [ $stage -le 6 ]; then
  datasets=libri
  model=LoResNet
  resnet_size=8
#  --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/spect/dev_wcmvn \
#  --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/spect/dev_wcmvn \
  for loss in soft asoft ; do
    echo -e "\n\033[1;4;31m Training ${model} with ${loss} in ${datasets}\033[0m\n"
    python TrainAndTest/Spectrogram/train_lores10_kaldi.py \
      --model ${model} \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/spect/dev_wcmvn \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/spect/test_wcmvn \
      --resnet-size ${resnet_size} \
      --nj 12 \
      --epochs 15 \
      --lr 0.1 \
      --milestones 7,11 \
      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/spect_wcmvn/${loss} \
      --resume Data/checkpoint/${model}${resnet_size}/${datasets}/spect_wcmvn/${loss}/checkpoint_1.pth \
      --kernel-size 5,5 \
      --channels 4,16,64 \
      --embedding-size 128 \
      --input-per-spks 256 \
      --num-valid 1 \
      --alpha 10 \
      --margin 0.4 \
      --s 30 \
      --m 3 \
      --loss-ratio 0.1 \
      --weight-decay 0.001 \
      --dropout-p 0.25 \
      --gpu-id 0 \
      --loss-type ${loss}

#    python TrainAndTest/Spectrogram/train_lores10_kaldi.py \
#      --model ${model} \
#      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/spect/dev_wcmvn \
#      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/libri/spect/test_wcmvn \
#      --resnet-size 8 \
#      --nj 12 \
#      --epochs 15 \
#      --lr 0.1 \
#      --milestones 7,11 \
#      --check-path Data/checkpoint/LoResNet8/${datasets}/spect_wcmvn/${loss}_norm_fc \
#      --resume Data/checkpoint/LoResNet8/${datasets}/spect_wcmvn/${loss}_norm_fc/checkpoint_1.pth \
#      --channels 4,16,64 \
#      --embedding-size 128 \
#      --input-per-spks 256 \
#      --num-valid 1 \
#      --alpha 10 \
#      --margin 0.4 \
#      --s 30 \
#      --m 3 \
#      --loss-ratio 0.05 \
#      --weight-decay 0.001 \
#      --dropout-p 0.25 \
#      --inst-norm \
#      --gpu-id 0 \
#      --loss-type ${loss}
  done
fi

if [ $stage -le 10 ]; then
  datasets=army
  model=LoResNet
  resnet_size=8
  for loss in soft; do
    python TrainAndTest/Spectrogram/train_lores10_kaldi.py \
      --model ${model} \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/army/dev_10k \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/army/test \
      --resnet-size 8 \
      --nj 12 \
      --epochs 15 \
      --lr 0.1 \
      --milestones 7,11 \
      --check-path Data/checkpoint/LoResNet8/${datasets}/spect_noc/${loss} \
      --resume Data/checkpoint/LoResNet8/${datasets}/spect_noc/${loss}/checkpoint_1.pth \
      --channels 4,16,64 \
      --embedding-size 128 \
      --input-per-spks 256 \
      --num-valid 1 \
      --alpha 10 \
      --margin 0.4 \
      --s 30 \
      --m 3 \
      --loss-ratio 0.05 \
      --weight-decay 0.001 \
      --dropout-p 0.25 \
      --inst-norm \
      --gpu-id 0 \
      --loss-type ${loss}
  done
fi

if [ $stage -le 15 ]; then
  datasets=army
  model=LoResNet
  resnet_size=8
  for loss in soft; do
    python TrainAndTest/Spectrogram/train_lores10_kaldi.py \
      --model ${model} \
      --train-dir /home/storage/yangwenhao/project/lstm_speaker_verification/data/army/spect/dev \
      --test-dir /home/storage/yangwenhao/project/lstm_speaker_verification/data/army/spect/test \
      --feat-format npy \
      --resnet-size 8 \
      --nj 10 \
      --epochs 15 \
      --lr 0.1 \
      --milestones 7,11 \
      --check-path Data/checkpoint/LoResNet8/${datasets}/spect_noc/${loss}_norm \
      --resume Data/checkpoint/LoResNet8/${datasets}/spect_noc/${loss}_norm/checkpoint_1.pth \
      --channels 4,16,64 \
      --embedding-size 128 \
      --input-per-spks 192 \
      --num-valid 1 \
      --alpha 10 \
      --margin 0.4 \
      --s 30 \
      --m 3 \
      --inst-norm \
      --loss-ratio 0.05 \
      --weight-decay 0.001 \
      --dropout-p 0.25 \
      --gpu-id 0 \
      --loss-type ${loss}
  done
fi

if [ $stage -le 60 ]; then
  lstm_dir=/home/work2020/yangwenhao/project/lstm_speaker_verification
  datasets=timit
  model=LoResNet
  resnet_size=8
  loss=coscenter
  encoder=None

  for loss_ratio in 0.5 0.1 0.05 0.01 ; do
    python TrainAndTest/Spectrogram/train_egs.py \
      --model ${model} \
      --train-dir ${lstm_dir}/data/${datasets}/egs/spect/train_log \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/spect/valid_log \
      --test-dir ${lstm_dir}/data/${datasets}/spect/test_log \
      --feat-format kaldi \
      --input-norm None \
      --resnet-size ${resnet_size} \
      --nj 10 \
      --epochs 12 \
      --lr 0.1 \
      --input-dim 161 \
      --milestones 6,10 \
      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/spect_egs_${encoder}/${loss}_3.5_dp05_${loss_ratio} \
      --resume Data/checkpoint/${model}${resnet_size}/${datasets}/spect_egs_${encoder}/${loss}_3.5_dp05_${loss_ratio}/checkpoint_24.pth \
      --channels 4,16,64 \
      --stride 2 \
      --embedding-size 128 \
      --optimizer sgd \
      --avg-size 4 \
      --time-dim 1 \
      --num-valid 2 \
      --alpha 10.8 \
      --margin 0.4 \
      --s 30 \
      --m 3 \
      --loss-ratio ${loss_ratio} \
      --weight-decay 0.001 \
      --dropout-p 0.5 \
      --gpu-id 0 \
      --cos-sim \
      --extract \
      --encoder-type ${encoder} \
      --loss-type ${loss}
  done
fi

exit 0;
