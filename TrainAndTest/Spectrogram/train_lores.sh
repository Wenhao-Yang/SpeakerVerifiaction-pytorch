#!/usr/bin/env bash

stage=50

waited=0
while [ `ps 8217 | wc -l` -eq 2 ]; do
  sleep 60
  waited=$(expr $waited + 1)
  echo -en "\033[1;4;31m Having waited for ${waited} minutes!\033[0m\r"
done

#stage=1
if [ $stage -le 0 ]; then
  for loss in soft ; do # 32,128,512; 8,32,128
    echo -e "\n\033[1;4;31m Training with ${loss} kernel 5x5\033[0m\n"
    python -W ignore TrainAndTest/Spectrogram/train_lores10_kaldi.py \
      --model LoResNet10 \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/dev_wcmvn \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/test_wcmvn \
      --nj 12 \
      --epochs 24 \
      --resnet-size 8 \
      --embedding-size 128 \
      --milestones 10,15,20 \
      --channels 64,128,256 \
      --check-path Data/checkpoint/LoResNet8/spect/${loss}_wcmvn \
      --resume Data/checkpoint/LoResNet8/spect/${loss}_wcmvn/checkpoint_20.pth \
      --loss-type ${loss} \
      --num-valid 2 \
      --dropout-p 0.25
  done
fi

#stage=100


if [ $stage -le 1 ]; then
#  for loss in center amsoft ; do/
  for loss in asoft amsoft center; do
    echo -e "\n\033[1;4;31m Finetuning with ${loss}\033[0m\n"
    python -W ignore TrainAndTest/Spectrogram/train_lores10_kaldi.py \
      --model LoResNet10 \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/dev_wcmvn \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/test_wcmvn \
      --nj 12 \
      --resnet-size 8 \
      --epochs 14 \
      --milestones 6,10 \
      --embedding-size 128 \
      --check-path Data/checkpoint/LoResNet10/spect/${loss}_wcmvn \
      --resume Data/checkpoint/LoResNet10/spect/soft_wcmvn/checkpoint_24.pth \
      --loss-type ${loss} \
      --loss-ratio 0.001 \
      --lr 0.01 \
      --lambda-max 1200 \
      --margin 0.35 \
      --s 15 \
      --m 3 \
      --num-valid 2 \
      --dropout-p 0.25
  done
fi

#stage=100
# kernel size trianing
if [ $stage -le 4 ]; then
  for kernel in '3,3' '3,7' '5,7' ; do
    echo -e "\n\033[1;4;31m Training with kernel size ${kernel} \033[0m\n"
    python TrainAndTest/Spectrogram/train_lores10_kaldi.py \
      --nj 12 \
      --epochs 18 \
      --milestones 8,13,18 \
      --resume Data/checkpoint/LoResNet10/spect/kernel_${kernel}/checkpoint_18.pth \
      --check-path Data/checkpoint/LoResNet10/spect/kernel_${kernel} \
      --kernel-size ${kernel}
  done

fi

if [ $stage -le 5 ]; then
  for loss in soft ; do
    echo -e "\n\033[1;4;31m Training with ${loss} kernel 3x3\033[0m\n"
    python TrainAndTest/Spectrogram/train_lores10_kaldi.py \
      --nj 12 \
      --epochs 10 \
      --milestones 4,7 \
      --check-path Data/checkpoint/LoResNet10/spect/${loss}_dp33_0.1 \
      --resume Data/checkpoint/LoResNet10/spect/${loss}_dp33/checkpoint_20.pth \
      --kernel-size 3,3 \
      --loss-type ${loss} \
      --num-valid 2 \
      --dropout-p 0.1

    echo -e "\n\033[1;4;31m Training with ${loss} kernel 5x5\033[0m\n"
    python TrainAndTest/Spectrogram/train_lores10_kaldi.py \
      --nj 12 \
      --epochs 10 \
      --milestones 4,7 \
      --check-path Data/checkpoint/LoResNet10/spect/${loss}_dp01 \
      --resume Data/checkpoint/LoResNet10/spect/${loss}/checkpoint_1.pth \
      --loss-type ${loss} \
      --num-valid 2 \
      --dropout-p 0.1
  done
fi

if [ $stage -le 6 ]; then
  for loss in soft ; do
    echo -e "\n\033[1;4;31m Continue Training with ${loss} kernel 3x3\033[0m\n"
    python TrainAndTest/Spectrogram/train_lores10_kaldi.py \
      --nj 12 \
      --epochs 4 \
      --resnet-size 10 \
      --check-path Data/checkpoint/LoResNet10/spectrogram/${loss} \
      --resume Data/checkpoint/LoResNet10/spectrogram/${loss}/checkpoint_20.pth \
      --channels 32,128,256,512 \
      --kernel-size 3,3 \
      --lr 0.0001 \
      --loss-type ${loss} \
      --num-valid 2 \
      --dropout-p 0.5
  done
fi

if [ $stage -le 7 ]; then
  for loss in soft ; do
    echo -e "\n\033[1;4;31m Training with ${loss} kernel 3x3\033[0m\n"
    python TrainAndTest/Spectrogram/train_lores10_kaldi.py \
      --nj 12 \
      --epochs 24 \
      --milestones 10,15,20 \
      --resnet-size 10 \
      --check-path Data/checkpoint/LoResNet10/spectrogram/${loss}_64 \
      --resume Data/checkpoint/LoResNet10/spectrogram/${loss}_64/checkpoint_20.pth \
      --channels 64,128,256,512 \
      --kernel-size 3,3 \
      --loss-type ${loss} \
      --num-valid 2 \
      --dropout-p 0.25
  done
fi


if [ $stage -le 15 ]; then
  for loss in soft ; do # 32,128,512; 8,32,128
    echo -e "\n\033[1;4;31m Training with ${loss} kernel 3,3\033[0m\n"
    python -W ignore TrainAndTest/Spectrogram/train_lores10_kaldi.py \
      --model LoResNet10 \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/dev_wcmvn \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/test_wcmvn \
      --input-per-spks 224 \
      --nj 12 \
      --epochs 24 \
      --resnet-size 18 \
      --embedding-size 128 \
      --kernel-size 3,3 \
      --avg-size 4 \
      --milestones 10,15,20 \
      --channels 64,128,256,256 \
      --check-path Data/checkpoint/LoResNet18/spect/${loss}_dp25 \
      --resume Data/checkpoint/LoResNet18/spect/${loss}_dp25/checkpoint_1.pth \
      --loss-type ${loss} \
      --lr 0.1 \
      --num-valid 2 \
      --gpu-id 1 \
      --dropout-p 0.25
  done

#  for loss in amsoft center asoft ; do # 32,128,512; 8,32,128
#    echo -e "\n\033[1;4;31m Training with ${loss} kernel 5x5\033[0m\n"
#    python -W ignore TrainAndTest/Spectrogram/train_lores10_kaldi.py \
#      --model LoResNet10 \
#      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/dev_wcmvn \
#      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/test_wcmvn \
#      --input-per-spks 224 \
#      --nj 12 \
#      --epochs 12 \
#      --resnet-size 10 \
#      --embedding-size 128 \
#      --kernel-size 3,3 \
#      --milestones 6,10 \
#      --channels 64,128,256,512 \
#      --check-path Data/checkpoint/LoResNet10/spect/${loss}_dp25 \
#      --resume Data/checkpoint/LoResNet10/spect/soft_dp25/checkpoint_24.pth \
#      --loss-type ${loss} \
#      --finetune \
#      --lr 0.01 \
#      --num-valid 2 \
#      --margin 0.35 \
#      --s 40 \
#      --m 4 \
#      --loss-ratio 0.01 \
#      --gpu-id 0 \
#      --dropout-p 0.25
#  done
fi

if [ $stage -le 20 ]; then
  dataset=aishell2

  for loss in soft ; do # 32,128,512; 8,32,128
    echo -e "\n\033[1;4;31m Training with ${loss} kernel 3,3\033[0m\n"
    python -W ignore TrainAndTest/Spectrogram/train_lores10_kaldi.py \
      --model LoResNet10 \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/${dataset}/spect/dev \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/${dataset}/spect/test \
      --input-per-spks 384 \
      --nj 12 \
      --epochs 24 \
      --resnet-size 18 \
      --embedding-size 128 \
      --kernel-size 3,3 \
      --avg-size 4 \
      --milestones 10,15,20 \
      --channels 64,128,256,256 \
      --check-path Data/checkpoint/LoResNet18/${dataset}/spect/${loss}_dp25 \
      --resume Data/checkpoint/LoResNet18/${dataset}/spect/${loss}_dp25/checkpoint_1.pth \
      --loss-type ${loss} \
      --lr 0.1 \
      --num-valid 2 \
      --gpu-id 1 \
      --dropout-p 0.25
  done

fi

if [ $stage -le 30 ]; then
  dataset=cnceleb
  model=LoResNet
  resnet_size=8
  for loss in soft ; do # 32,128,512; 8,32,128
    echo -e "\n\033[1;4;31m Training with ${loss} kernel 5,5\033[0m\n"
    python -W ignore TrainAndTest/Spectrogram/train_lores10_kaldi.py \
      --model LoResNet \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/${dataset}/spect/dev_power \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/${dataset}/spect/test_power \
      --input-per-spks 224 \
      --feat-format npy \
      --nj 12 \
      --epochs 24 \
      --resnet-size ${resnet_size} \
      --embedding-size 128 \
      --avg-size 4 \
      --milestones 10,15,20 \
      --channels 64,128,256 \
      --check-path Data/checkpoint/LoResNet8/${dataset}/spect/${loss}_dp25 \
      --resume Data/checkpoint/LoResNet8/${dataset}/spect/${loss}_dp25/checkpoint_1.pth \
      --loss-type ${loss} \
      --lr 0.1 \
      --num-valid 2 \
      --gpu-id 0 \
      --dropout-p 0.25
  done

fi

if [ $stage -le 31 ]; then
  dataset=vox1
  model=GradResNet
  resnet_size=8
  for loss in soft; do # 32,128,512; 8,32,128
    echo -e "\n\033[1;4;31m Training with ${loss} kernel 5,5\033[0m\n"
    python -W ignore TrainAndTest/Spectrogram/train_lores10_kaldi.py \
      --model ${model} \
      --train-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/${dataset}/spect/dev_power_32 \
      --test-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/${dataset}/spect/test_power_32 \
      --input-per-spks 224 \
      --feat-format kaldi \
      --nj 10 \
      --epochs 12 \
      --resnet-size ${resnet_size} \
      --embedding-size 128 \
      --avg-size 4 \
      --alpha 12 \
      --inst-norm \
      --batch-size 128 \
      --milestones 3,7,10 \
      --channels 64,128,256 \
      --check-path Data/checkpoint/GradResNet8_mean/${dataset}_power_spk/spect_time/${loss}_dp25 \
      --resume Data/checkpoint/GradResNet8_mean/${dataset}_power_spk/spect_time/soft_dp25/checkpoint_24.pth \
      --loss-type ${loss} \
      --finetune \
      --margin 0.3 \
      --s 30 \
      --loss-ratio 0.1 \
      --lr 0.01 \
      --num-valid 2 \
      --gpu-id 0 \
      --cos-sim \
      --dropout-p 0.25
  done

fi

#stage=50
if [ $stage -le 40 ]; then
  datasets=all_army
  model=LoResNet
  resnet_size=8
  for loss in soft; do
    python TrainAndTest/Spectrogram/train_lores10_kaldi.py \
      --model ${model} \
      --train-dir /home/storage/yangwenhao/project/lstm_speaker_verification/data/all_army/spect/dev \
      --test-dir /home/storage/yangwenhao/project/lstm_speaker_verification/data/all_army/spect/test \
      --feat-format npy \
      --resnet-size 8 \
      --nj 10 \
      --epochs 15 \
      --lr 0.1 \
      --milestones 7,11 \
      --check-path Data/checkpoint/LoResNet8/${datasets}/spect_noc/${loss} \
      --resume Data/checkpoint/LoResNet8/${datasets}/spect_noc/${loss}/checkpoint_1.pth \
      --channels 64,128,256 \
      --embedding-size 128 \
      --input-per-spks 192 \
      --num-valid 1 \
      --alpha 12 \
      --margin 0.4 \
      --s 30 \
      --m 3 \
      --loss-ratio 0.05 \
      --weight-decay 0.001 \
      --dropout-p 0.25 \
      --gpu-id 0 \
      --loss-type ${loss}
  done
fi

#stage=50
if [ $stage -le 50 ]; then
  lstm_dir=/home/work2020/yangwenhao/project/lstm_speaker_verification
  datasets=vox1
  model=LoResNet
  resnet_size=8
  for loss in soft arcsoft ; do
    echo -e "\n\033[1;4;31m Training ${model} in vox1_egs with ${loss} with mean normalization \033[0m\n"
    python TrainAndTest/Spectrogram/train_egs.py \
      --model ${model} \
      --train-dir ${lstm_dir}/data/vox1/egs/spect/dev_log \
      --valid-dir ${lstm_dir}/data/vox1/egs/spect/valid_log \
      --test-dir ${lstm_dir}/data/vox1/spect/test_log \
      --feat-format kaldi \
      --input-norm Mean \
      --resnet-size ${resnet_size} \
      --nj 10 \
      --epochs 20 \
      --lr 0.1 \
      --milestones 5,10,15 \
      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/spect_egs/${loss}_dp25 \
      --resume Data/checkpoint/${model}${resnet_size}/${datasets}/spect_egs/${loss}_dp25/checkpoint_24.pth \
      --channels 64,128,256 \
      --batch-size 128 \
      --embedding-size 128 \
      --avg-size 4 \
      --num-center 2 \
      --num-valid 2 \
      --alpha 12 \
      --margin 0.3 \
      --s 15 \
      --m 3 \
      --loss-ratio 0.01 \
      --weight-decay 0.001 \
      --dropout-p 0.25 \
      --gpu-id 0 \
      --extract \
      --cos-sim \
      --loss-type ${loss}
  done

  resnet_size=10
  for loss in soft ; do
    echo -e "\n\033[1;4;31m Training ${model} in vox1_egs with ${loss} with mean normalization \033[0m\n"
    python TrainAndTest/Spectrogram/train_egs.py \
      --model ${model} \
      --train-dir ${lstm_dir}/data/vox1/egs/spect/dev_log \
      --valid-dir ${lstm_dir}/data/vox1/egs/spect/valid_log \
      --test-dir ${lstm_dir}/data/vox1/spect/test_log \
      --feat-format kaldi \
      --input-norm Mean \
      --resnet-size ${resnet_size} \
      --nj 10 \
      --epochs 20 \
      --lr 0.1 \
      --milestones 5,10,15 \
      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/spect_egs/${loss}_dp25 \
      --resume Data/checkpoint/${model}${resnet_size}/${datasets}/spect_egs/${loss}_dp25/checkpoint_24.pth \
      --channels 64,128,256,512 \
      --batch-size 128 \
      --embedding-size 128 \
      --avg-size 1 \
      --num-valid 2 \
      --alpha 12 \
      --margin 0.3 \
      --s 15 \
      --m 3 \
      --loss-ratio 0.01 \
      --weight-decay 0.001 \
      --dropout-p 0.25 \
      --gpu-id 0 \
      --extract \
      --cos-sim \
      --loss-type ${loss}
  done

fi
stage=1000
if [ $stage -le 51 ]; then
  lstm_dir=/home/work2020/yangwenhao/project/lstm_speaker_verification
  datasets=cnceleb
  model=GradResNet
  resnet_size=8
  for loss in soft; do
#    python TrainAndTest/Spectrogram/train_egs.py \
#      --model ${model} \
#      --train-dir ${lstm_dir}/data/${datasets}/egs/spect/dev_4w \
#      --valid-dir ${lstm_dir}/data/${datasets}/egs/spect/valid_4w \
#      --test-dir ${lstm_dir}/data/${datasets}/spect/test \
#      --feat-format kaldi \
#      --inst-norm \
#      --resnet-size ${resnet_size} \
#      --nj 10 \
#      --epochs 24 \
#      --lr 0.1 \
#      --input-dim 161 \
#      --milestones 10,15,20 \
#      --check-path Data/checkpoint/${model}8/${datasets}_4w/spect_egs/${loss}_dp25 \
#      --resume Data/checkpoint/${model}8/${datasets}_4w/spect_egs/${loss}_dp25/checkpoint_24.pth \
#      --channels 16,64,128 \
#      --embedding-size 128 \
#      --avg-size 4 \
#      --num-valid 2 \
#      --alpha 12 \
#      --margin 0.4 \
#      --s 30 \
#      --m 3 \
#      --loss-ratio 0.05 \
#      --weight-decay 0.001 \
#      --dropout-p 0.25 \
#      --gpu-id 0 \
#      --cos-sim \
#      --extract \
#      --loss-type ${loss}

#    python TrainAndTest/Spectrogram/train_egs.py \
#      --model ${model} \
#      --train-dir ${lstm_dir}/data/${datasets}/egs/spect/dev_4w \
#      --valid-dir ${lstm_dir}/data/${datasets}/egs/spect/valid_4w \
#      --test-dir ${lstm_dir}/data/${datasets}/spect/test \
#      --feat-format kaldi \
#      --inst-norm \
#      --resnet-size ${resnet_size} \
#      --nj 10 \
#      --epochs 24 \
#      --lr 0.1 \
#      --milestones 10,15,20 \
#      --input-dim 161 \
#      --check-path Data/checkpoint/${model}8/${datasets}_4w/spect_egs_vad/${loss}_dp25 \
#      --resume Data/checkpoint/${model}8/${datasets}_4w/spect_egs_vad/${loss}_dp25/checkpoint_24.pth \
#      --channels 16,64,128 \
#      --embedding-size 128 \
#      --avg-size 4 \
#      --num-valid 2 \
#      --alpha 12 \
#      --margin 0.4 \
#      --s 30 \
#      --m 3 \
#      --loss-ratio 0.05 \
#      --weight-decay 0.001 \
#      --dropout-p 0.25 \
#      --gpu-id 0 \
#      --cos-sim \
#      --vad \
#      --extract \
#      --loss-type ${loss}
#
        python TrainAndTest/Spectrogram/train_egs.py \
      --model ${model} \
      --train-dir ${lstm_dir}/data/${datasets}/egs/spect/dev_4w \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/spect/valid_4w \
      --test-dir ${lstm_dir}/data/${datasets}/spect/test \
      --feat-format kaldi \
      --inst-norm \
      --resnet-size ${resnet_size} \
      --nj 10 \
      --epochs 24 \
      --lr 0.1 \
      --milestones 10,15,20 \
      --input-dim 161 \
      --check-path Data/checkpoint/${model}8/${datasets}_4w/spect_egs_ince4/${loss}_dp25 \
      --resume Data/checkpoint/${model}8/${datasets}_4w/spect_egs_ince4/${loss}_dp25/checkpoint_24.pth \
      --channels 16,64,128 \
      --embedding-size 128 \
      --avg-size 4 \
      --num-valid 2 \
      --alpha 12 \
      --margin 0.4 \
      --s 30 \
      --m 3 \
      --loss-ratio 0.05 \
      --weight-decay 0.001 \
      --dropout-p 0.25 \
      --gpu-id 0 \
      --cos-sim \
      --inception \
      --extract \
      --loss-type ${loss}
  done
fi

if [ $stage -le 52 ]; then
  lstm_dir=/home/work2020/yangwenhao/project/lstm_speaker_verification
  datasets=cnceleb
  model=TimeFreqResNet
  resnet_size=8
  for loss in soft; do
    python TrainAndTest/Spectrogram/train_egs.py \
      --model ${model} \
      --train-dir ${lstm_dir}/data/${datasets}/egs/spect/dev_4w \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/spect/valid_4w \
      --test-dir ${lstm_dir}/data/${datasets}/spect/test \
      --feat-format kaldi \
      --inst-norm \
      --resnet-size ${resnet_size} \
      --nj 10 \
      --epochs 24 \
      --lr 0.1 \
      --input-dim 161 \
      --milestones 10,15,20 \
      --check-path Data/checkpoint/${model}8/${datasets}_4w/spect_egs/${loss}_dp25 \
      --resume Data/checkpoint/${model}8/${datasets}_4w/spect_egs/${loss}_dp25/checkpoint_24.pth \
      --channels 16,64,128 \
      --embedding-size 128 \
      --avg-size 4 \
      --num-valid 2 \
      --alpha 12 \
      --margin 0.4 \
      --s 30 \
      --m 3 \
      --loss-ratio 0.05 \
      --weight-decay 0.001 \
      --dropout-p 0.25 \
      --gpu-id 0 \
      --cos-sim \
      --extract \
      --loss-type ${loss}
  done
fi

if [ $stage -le 60 ]; then
  lstm_dir=/home/work2020/yangwenhao/project/lstm_speaker_verification
  datasets=timit
  model=TimeFreqResNet
  resnet_size=8

  for loss in asoft; do
    python TrainAndTest/Spectrogram/train_egs.py \
      --model ${model} \
      --train-dir ${lstm_dir}/data/${datasets}/egs/spect/train_power \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/spect/valid_power\
      --test-dir ${lstm_dir}/data/${datasets}/spect/test_power \
      --feat-format kaldi \
      --resnet-size ${resnet_size} \
      --nj 10 \
      --epochs 12 \
      --lr 0.1 \
      --input-dim 161 \
      --milestones 6,10 \
      --check-path Data/checkpoint/${model}8/${datasets}/spect_egs/${loss}_dp25 \
      --resume Data/checkpoint/${model}8/${datasets}/spect_egs/${loss}_dp25/checkpoint_24.pth \
      --channels 4,16,64 \
      --embedding-size 128 \
      --avg-size 4 \
      --num-valid 2 \
      --alpha 12 \
      --margin 0.4 \
      --s 30 \
      --m 3 \
      --loss-ratio 0.05 \
      --weight-decay 0.001 \
      --dropout-p 0.25 \
      --gpu-id 0 \
      --cos-sim \
      --extract \
      --loss-type ${loss}
  done

  for loss in soft; do
    python TrainAndTest/Spectrogram/train_egs.py \
      --model ${model} \
      --train-dir ${lstm_dir}/data/${datasets}/egs/spect/train_power \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/spect/valid_power\
      --test-dir ${lstm_dir}/data/${datasets}/spect/test_power \
      --feat-format kaldi \
      --resnet-size ${resnet_size} \
      --nj 10 \
      --epochs 12 \
      --lr 0.1 \
      --input-dim 161 \
      --milestones 6,10 \
      --check-path Data/checkpoint/${model}8/${datasets}/spect_egs_chn8/${loss}_dp25 \
      --resume Data/checkpoint/${model}8/${datasets}/spect_egs_chn8/${loss}_dp25/checkpoint_24.pth \
      --channels 4,8,64 \
      --embedding-size 128 \
      --avg-size 4 \
      --num-valid 2 \
      --alpha 12 \
      --margin 0.4 \
      --s 30 \
      --m 3 \
      --loss-ratio 0.05 \
      --weight-decay 0.001 \
      --dropout-p 0.25 \
      --gpu-id 0 \
      --cos-sim \
      --extract \
      --loss-type ${loss}
  done

fi
if [ $stage -le 61 ]; then
  lstm_dir=/home/work2020/yangwenhao/project/lstm_speaker_verification
  datasets=vox1
  model=LoResNet
  resnet_size=8
  for loss in soft arcsoft ; do
    python TrainAndTest/Spectrogram/train_egs.py \
      --model ${model} \
      --train-dir ${lstm_dir}/data/${datasets}/egs/spect/train_power \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/spect/valid_power \
      --test-dir ${lstm_dir}/data/${datasets}/spect/test_power \
      --log-scale \
      --feat-format kaldi \
      --resnet-size ${resnet_size} \
      --nj 10 \
      --epochs 12 \
      --lr 0.1 \
      --input-dim 161 \
      --milestones 6,10 \
      --check-path Data/checkpoint/${model}8/${datasets}/spect_egs_log/${loss}_dp05 \
      --resume Data/checkpoint/${model}8/${datasets}/spect_egs_log/${loss}_dp05/checkpoint_12.pth \
      --alpha 12 \
      --channels 4,16,64 \
      --embedding-size 128 \
      --avg-size 4 \
      --num-valid 2 \
      --margin 0.4 \
      --s 30 \
      --m 3 \
      --loss-ratio 0.05 \
      --weight-decay 0.001 \
      --dropout-p 0.5 \
      --gpu-id 0 \
      --cos-sim \
      --extract \
      --loss-type ${loss}
  done
fi

stage=100
if [ $stage -le 62 ]; then
  lstm_dir=/home/work2020/yangwenhao/project/lstm_speaker_verification
  datasets=army
  model=LoResNet
  resnet_size=10
  for loss in soft ; do
    echo -e "\n\033[1;4;31m Training LoResNet in vox1 with ${loss} kernel 5,5 \033[0m\n"
    python TrainAndTest/Spectrogram/train_egs.py \
      --model ${model} \
      --train-dir ${lstm_dir}/data/${datasets}/egs/spect/dev_v1 \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/spect/valid_v1 \
      --test-dir ${lstm_dir}/data/${datasets}/spect/test_8k \
      --feat-format kaldi \
      --resnet-size ${resnet_size} \
      --inst-norm \
      --batch-size 256 \
      --nj 12 \
      --epochs 20 \
      --lr 0.1 \
      --input-dim 81 \
      --milestones 5,10,15 \
      --padding 0,0 \
      --check-path Data/checkpoint/${model}10/${datasets}_v1/spect_egs_pad0/${loss}_dp01 \
      --resume Data/checkpoint/${model}10/${datasets}_v1/spect_egs_pad0/soft_dp01/checkpoint_24.pth \
      --channels 64,128,256,256 \
      --embedding-size 128 \
      --avg-size 4 \
      --num-valid 4 \
      --alpha 12 \
      --margin 0.3 \
      --s 30 \
      --m 3 \
      --loss-ratio 0.01 \
      --weight-decay 0.001 \
      --dropout-p 0.1 \
      --gpu-id 0 \
      --cos-sim \
      --extract \
      --loss-type ${loss}
  done
#  python TrainAndTest/Spectrogram/train_egs.py \
#      --model ${model} \
#      --train-dir ${lstm_dir}/data/${datasets}/egs/spect/dev_v1 \
#      --valid-dir ${lstm_dir}/data/${datasets}/egs/spect/valid_v1 \
#      --test-dir ${lstm_dir}/data/${datasets}/spect/test_8k \
#      --feat-format kaldi \
#      --resnet-size ${resnet_size} \
#      --inst-norm \
#      --batch-size 256 \
#      --nj 12 \
#      --epochs 20 \
#      --lr 0.1 \
#      --input-dim 81 \
#      --milestones 5,10,15 \
#      --check-path Data/checkpoint/${model}10/${datasets}_v1/spect_egs/soft_dp25 \
#      --resume Data/checkpoint/${model}10/${datasets}_v1/spect_egs/soft_dp25/checkpoint_24.pth \
#      --channels 64,128,256,256 \
#      --embedding-size 128 \
#      --avg-size 4 \
#      --num-valid 4 \
#      --alpha 12 \
#      --margin 0.3 \
#      --s 30 \
#      --m 3 \
#      --loss-ratio 0.05 \
#      --weight-decay 0.001 \
#      --dropout-p 0.25 \
#      --gpu-id 0 \
#      --cos-sim \
#      --extract \
#      --loss-type soft
fi