#!/usr/bin/env bash

stage=61

waited=0
while [ `ps 32449 | wc -l` -eq 2 ]; do
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
  encod=None
  transform=None

  for loss in asoft ; do
    echo -e "\n\033[1;4;31m Training ${model} in vox1_egs with ${loss} with mean normalization \033[0m\n"
    python TrainAndTest/Spectrogram/train_egs.py \
      --model ${model} \
      --train-dir ${lstm_dir}/data/vox1/egs/spect/dev_log \
      --train-test-dir ${lstm_dir}/data/vox1/spect/dev_log \
      --valid-dir ${lstm_dir}/data/vox1/egs/spect/valid_log \
      --test-dir ${lstm_dir}/data/vox1/spect/test_log \
      --train-trials trials_2w \
      --feat-format kaldi \
      --input-norm Mean \
      --transform None \
      --resnet-size ${resnet_size} \
      --nj 10 \
      --epochs 1 \
      --lr 0.1 \
      --milestones 5,10,15 \
      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/spect_egs_${encod}/${loss}_dp25_${transform} \
      --resume Data/checkpoint/${model}${resnet_size}/${datasets}/spect_egs_${encod}/${loss}_dp25_${transform}/checkpoint_20.pth \
      --channels 64,128,256 \
      --kernel-size 5,5 \
      --stride 2 \
      --batch-size 128 \
      --embedding-size 128 \
      --avg-size 4 \
      --time-dim 1 \
      --num-valid 2 \
      --alpha 12 \
      --margin 0.3 \
      --s 15 \
      --m 3 \
      --loss-ratio 0.01 \
      --weight-decay 0.001 \
      --dropout-p 0.25 \
      --gpu-id 0 \
      --encoder-type ${encod} \
      --cos-sim \
      --loss-type ${loss}
  done

#  resnet_size=10
#  for loss in soft ; do
#    echo -e "\n\033[1;4;31m Training ${model} in vox1_egs with ${loss} with mean normalization \033[0m\n"
#    python TrainAndTest/Spectrogram/train_egs.py \
#      --model ${model} \
#      --train-dir ${lstm_dir}/data/vox1/egs/spect/dev_log \
#      --valid-dir ${lstm_dir}/data/vox1/egs/spect/valid_log \
#      --test-dir ${lstm_dir}/data/vox1/spect/test_log \
#      --feat-format kaldi \
#      --input-norm Mean \
#      --resnet-size ${resnet_size} \
#      --nj 10 \
#      --epochs 20 \
#      --lr 0.1 \
#      --milestones 5,10,15 \
#      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/spect_egs/${loss}_dp25 \
#      --resume Data/checkpoint/${model}${resnet_size}/${datasets}/spect_egs/${loss}_dp25/checkpoint_24.pth \
#      --channels 64,128,256,512 \
#      --batch-size 128 \
#      --embedding-size 128 \
#      --avg-size 1 \
#      --num-valid 2 \
#      --alpha 12 \
#      --margin 0.3 \
#      --s 15 \
#      --m 3 \
#      --loss-ratio 0.01 \
#      --weight-decay 0.001 \
#      --dropout-p 0.25 \
#      --gpu-id 0 \
#      --extract \
#      --cos-sim \
#      --loss-type ${loss}
#  done

fi

#stage=6300

if [ $stage -le 51 ]; then
  lstm_dir=/home/work2020/yangwenhao/project/lstm_speaker_verification
  model=ThinResNet
  resnet_size=34
  datasets=vox1
  feat=spect_161
  loss=soft
  encod=STAP

  for filter in None ; do
    echo -e "\n\033[1;4;31m Training ${model}_${encod} with ${loss}\033[0m\n"
    python -W ignore TrainAndTest/Spectrogram/train_egs.py \
      --train-dir ${lstm_dir}/data/vox1/egs/spect/dev_log \
      --train-test-dir ${lstm_dir}/data/vox1/spect/dev_log \
      --valid-dir ${lstm_dir}/data/vox1/egs/spect/valid_log \
      --test-dir ${lstm_dir}/data/vox1/spect/test_log \
      --train-trials trials_2w \
      --nj 10 \
      --epochs 22 \
      --milestones 8,13,18 \
      --model ${model} \
      --resnet-size ${resnet_size} \
      --stride 1 \
      --fast \
      --feat-format kaldi \
      --filter ${filter} \
      --embedding-size 128 \
      --batch-size 128 \
      --accu-steps 1 \
      --feat-dim 161 \
      --time-dim 1 \
      --avg-size 1 \
      --kernel-size 5,5 \
      --lr 0.1 \
      --encoder-type ${encod} \
      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/${feat}_${encod}_fast/${loss}_${filter} \
      --resume Data/checkpoint/${model}${resnet_size}/${datasets}/${feat}_${encod}_fast/${loss}_${filter}/checkpoint_9.pth \
      --input-per-spks 384 \
      --cos-sim \
      --veri-pairs 9600 \
      --gpu-id 0 \
      --num-valid 2 \
      --loss-type soft \
      --remove-vad
  done
fi

#stage=6300
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
  model=LoResNet
  resnet_size=8
  loss=soft

  for encoder in None ; do
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
      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/spect_egs_${encoder}/${loss}_dp05_clip_0001 \
      --resume Data/checkpoint/${model}${resnet_size}/${datasets}/spect_egs_${encoder}/${loss}_dp05_clip_0001/checkpoint_24.pth \
      --channels 4,16,64 \
      --stride 1 \
      --embedding-size 128 \
      --optimizer sgd \
      --avg-size 4 \
      --time-dim 1 \
      --num-valid 2 \
      --alpha 10.8 \
      --margin 0.4 \
      --s 30 \
      --m 3 \
      --loss-ratio 0.05 \
      --weight-decay 0.0001 \
      --dropout-p 0.5 \
      --gpu-id 0 \
      --cos-sim \
      --extract \
      --encoder-type ${encoder} \
      --loss-type ${loss}
  done

#  for loss in soft; do
#    python TrainAndTest/Spectrogram/train_egs.py \
#      --model ${model} \
#      --train-dir ${lstm_dir}/data/${datasets}/egs/spect/train_power \
#      --valid-dir ${lstm_dir}/data/${datasets}/egs/spect/valid_power\
#      --test-dir ${lstm_dir}/data/${datasets}/spect/test_power \
#      --feat-format kaldi \
#      --resnet-size ${resnet_size} \
#      --nj 10 \
#      --epochs 12 \
#      --lr 0.1 \
#      --input-dim 161 \
#      --milestones 6,10 \
#      --check-path Data/checkpoint/${model}8/${datasets}/spect_egs_chn8/${loss}_dp25 \
#      --resume Data/checkpoint/${model}8/${datasets}/spect_egs_chn8/${loss}_dp25/checkpoint_24.pth \
#      --channels 4,8,64 \
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
#  done

fi

#stage=10000
if [ $stage -le 61 ]; then
  lstm_dir=/home/work2020/yangwenhao/project/lstm_speaker_verification
  datasets=vox1
  model=LoResNet
  resnet_size=8
  mask_layer=time_freq
  loss=soft

  python TrainAndTest/Spectrogram/train_egs.py \
    --model ${model} \
    --train-dir ${lstm_dir}/data/vox1/egs/spect/dev_log \
    --train-test-dir ${lstm_dir}/data/vox1/spect/dev_log/trials_dir \
    --valid-dir ${lstm_dir}/data/vox1/egs/spect/valid_log \
    --test-dir ${lstm_dir}/data/vox1/spect/test_log \
    --train-trials trials_2w \
    --input-norm Mean \
    --feat-format kaldi \
    --resnet-size ${resnet_size} \
    --nj 10 \
    --epochs 20 \
    --lr 0.1 \
    --input-dim 161 \
    --milestones 5,10,15 \
    --check-path Data/checkpoint/${model}8/${datasets}/spect_egs_${mask_layer}/${loss}_dp25 \
    --resume Data/checkpoint/${model}8/${datasets}/spect_egs_${mask_layer}/${loss}_dp25/checkpoint_12.pth \
    --alpha 12 \
    --mask-layer ${mask_layer} \
    --channels 64,128,256 \
    --embedding-size 128 \
    --time-dim 1 \
    --avg-size 4 \
    --num-valid 2 \
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


  for block_type in seblock cbam ; do
    python TrainAndTest/Spectrogram/train_egs.py \
      --model ${model} \
      --train-dir ${lstm_dir}/data/vox1/egs/spect/dev_log \
      --train-test-dir ${lstm_dir}/data/vox1/spect/dev_log/trials_dir \
      --valid-dir ${lstm_dir}/data/vox1/egs/spect/valid_log \
      --test-dir ${lstm_dir}/data/vox1/spect/test_log \
      --train-trials trials_2w \
      --input-norm Mean \
      --feat-format kaldi \
      --resnet-size ${resnet_size} \
      --nj 10 \
      --epochs 20 \
      --lr 0.1 \
      --input-dim 161 \
      --milestones 5,10,15 \
      --check-path Data/checkpoint/${model}8/${datasets}/spect_egs_${block_type}/${loss}_dp25 \
      --resume Data/checkpoint/${model}8/${datasets}/spect_egs_${block_type}/${loss}_dp25/checkpoint_12.pth \
      --alpha 12 \
      --block-type ${block_type} \
      --channels 64,128,256 \
      --embedding-size 128 \
      --time-dim 1 \
      --avg-size 4 \
      --num-valid 2 \
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

stage=10000
if [ $stage -le 62 ]; then
  lstm_dir=/home/work2020/yangwenhao/project/lstm_speaker_verification
  datasets=army
  model=LoResNet
  resnet_size=10
  loss=soft
  for encod in None ; do
    echo -e "\n\033[1;4;31m Training LoResNet${resnet_size} in ${datasets} with ${loss} kernel 5,5 \033[0m\n"
    python TrainAndTest/Spectrogram/train_egs.py \
      --model ${model} \
      --train-dir ${lstm_dir}/data/${datasets}/egs/spect/dev_v2 \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/spect/valid_v2 \
      --test-dir ${lstm_dir}/data/${datasets}/spect/test_8k \
      --feat-format kaldi \
      --resnet-size ${resnet_size} \
      --input-norm Mean \
      --batch-size 256 \
      --nj 12 \
      --epochs 20 \
      --lr 0.1 \
      --input-dim 81 \
      --milestones 5,10,15 \
      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}_v2/spect_egs_fast_${encod}/${loss}_dp01 \
      --resume Data/checkpoint/${model}${resnet_size}/${datasets}_v2/spect_egs_fast_${encod}/soft_dp01/checkpoint_24.pth \
      --channels 32,64,128,256 \
      --embedding-size 128 \
      --encoder-type ${encod} \
      --avg-size 4 \
      --time-dim 1 \
      --stride 1 \
      --fast \
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
#stage=10000
if [ $stage -le 63 ]; then
  lstm_dir=/home/work2020/yangwenhao/project/lstm_speaker_verification
  datasets=army
  model=MultiResNet
  resnet_size=8
  loss=soft
  encod=None
  transform=None
  for loss in soft ; do
    echo -e "\n\033[1;4;31m Training ${model} in vox1 with ${loss} kernel 5,5 \033[0m\n"
    python TrainAndTest/Spectrogram/train_egs_multi.py \
      --model ${model} \
      --train-dir-a ${lstm_dir}/data/${datasets}/egs/spect/aishell2_dev_8k_v2 \
      --train-dir-b ${lstm_dir}/data/${datasets}/egs/spect/vox1_dev_8k_v2 \
      --valid-dir-a ${lstm_dir}/data/${datasets}/egs/spect/aishell2_valid_8k_v2 \
      --valid-dir-b ${lstm_dir}/data/${datasets}/egs/spect/vox1_valid_8k_v2 \
      --test-dir ${lstm_dir}/data/${datasets}/spect/test_8k \
      --feat-format kaldi \
      --resnet-size ${resnet_size} \
      --input-norm Mean \
      --batch-size 192 \
      --nj 12 \
      --epochs 24 \
      --lr 0.1 \
      --input-dim 81 \
      --stride 2 \
      --milestones 8,14,20 \
      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}_x2/spect_egs_${encod}/${loss}_dp25_b192_${transform} \
      --resume Data/checkpoint/${model}${resnet_size}/${datasets}_x2/spect_egs_${encod}/soft_dp25_b192_${transform}/checkpoint_24.pth \
      --channels 64,128,256 \
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
      --loss-ratio 0.6 \
      --grad-clip 0 \
      --weight-decay 0.001 \
      --dropout-p 0.25 \
      --gpu-id 0 \
      --cos-sim \
      --extract \
      --loss-type ${loss}
  done
fi

#stage=10000
if [ $stage -le 64 ]; then
  lstm_dir=/home/work2020/yangwenhao/project/lstm_speaker_verification
  datasets=army
  model=MultiResNet
  resnet_size=10
  loss=soft
  encod=None
  transform=None
  loss_ratio=0.01
  alpha=13
  for loss in soft ; do
    echo -e "\n\033[1;4;31m Training ${model}_${resnet_size} in vox1 with ${loss} kernel 5,5 \033[0m\n"

    python TrainAndTest/Spectrogram/train_egs_multi.py \
      --model ${model} \
      --train-dir-a ${lstm_dir}/data/${datasets}/egs/spect/aishell2_dev_8k_v2 \
      --train-dir-b ${lstm_dir}/data/${datasets}/egs/spect/vox1_dev_8k_v2 \
      --train-test-dir ${lstm_dir}/data/${datasets}/spect/dev_8k_v2/trials_dir \
      --valid-dir-a ${lstm_dir}/data/${datasets}/egs/spect/aishell2_valid_8k_v2 \
      --valid-dir-b ${lstm_dir}/data/${datasets}/egs/spect/vox1_valid_8k_v2 \
      --test-dir ${lstm_dir}/data/${datasets}/spect/test_8k \
      --feat-format kaldi \
      --resnet-size ${resnet_size} \
      --input-norm Mean \
      --batch-size 256 \
      --nj 12 \
      --epochs 7 \
      --lr 0.001 \
      --input-dim 81 \
      --fast \
      --stride 1 \
      --milestones 3 \
      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}_x2/spect_egs_${encod}/${loss}_dp25_b256_${alpha}_fast \
      --resume Data/checkpoint/${model}${resnet_size}/${datasets}_x2/spect_egs_${encod}/${loss}_dp25_b256_${alpha}_fast/checkpoint_17.pth \
      --channels 16,64,128,256 \
      --embedding-size 128 \
      --transform ${transform} \
      --encoder-type ${encod} \
      --time-dim 1 \
      --avg-size 4 \
      --num-valid 4 \
      --alpha ${alpha} \
      --margin 0.3 \
      --s 30 \
      --m 3 \
      --loss-ratio ${loss_ratio} \
      --set-ratio 1.0 \
      --grad-clip 0 \
      --weight-decay 0.001 \
      --dropout-p 0.25 \
      --gpu-id 0 \
      --cos-sim \
      --extract \
      --loss-type ${loss}

  done
fi

if [ $stage -le 65 ]; then
  lstm_dir=/home/work2020/yangwenhao/project/lstm_speaker_verification
  datasets=army
  model=MultiResNet
  resnet_size=10
  loss=soft
  encod=None
  transform=None
  loss_ratio=0.01
  alpha=13
  for loss in soft ; do
    echo -e "\n\033[1;4;31m Training ${model}_${resnet_size} in vox1 with ${loss} kernel 5,5 \033[0m\n"

    python TrainAndTest/Spectrogram/train_egs_multi.py \
      --model ${model} \
      --train-dir-a ${lstm_dir}/data/${datasets}/egs/spect/aishell2_dev_8k_v4 \
      --train-dir-b ${lstm_dir}/data/${datasets}/egs/spect/vox_dev_8k_v4 \
      --train-test-dir ${lstm_dir}/data/${datasets}/spect/dev_8k_v2/trials_dir \
      --valid-dir-a ${lstm_dir}/data/${datasets}/egs/spect/aishell2_valid_8k_v4 \
      --valid-dir-b ${lstm_dir}/data/${datasets}/egs/spect/vox_valid_8k_v4 \
      --test-dir ${lstm_dir}/data/${datasets}/spect/test_8k \
      --feat-format kaldi \
      --resnet-size ${resnet_size} \
      --input-norm Mean \
      --batch-size 256 \
      --nj 10 \
      --epochs 24 \
      --lr 0.1 \
      --input-dim 81 \
      --fast \
      --stride 1 \
      --milestones 8,14,20 \
      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}_x4/spect_egs_${encod}/${loss}_dp25_b256_${alpha}_fast \
      --resume Data/checkpoint/${model}${resnet_size}/${datasets}_x4/spect_egs_${encod}/${loss}_dp25_b256_${alpha}_fast/checkpoint_29.pth \
      --channels 16,64,128,256 \
      --embedding-size 128 \
      --transform ${transform} \
      --encoder-type ${encod} \
      --time-dim 1 \
      --avg-size 4 \
      --num-valid 4 \
      --alpha ${alpha} \
      --margin 0.3 \
      --s 30 \
      --m 3 \
      --loss-ratio ${loss_ratio} \
      --set-ratio 1.0 \
      --grad-clip 0 \
      --weight-decay 0.001 \
      --dropout-p 0.25 \
      --gpu-id 0 \
      --cos-sim \
      --extract \
      --loss-type ${loss}
  done
fi

stage=10000
if [ $stage -le 80 ]; then
  lstm_dir=/home/work2020/yangwenhao/project/lstm_speaker_verification
  datasets=vox2
  model=LoResNet
  resnet_size=10
  for loss in soft ; do
    echo -e "\n\033[1;4;31m Training ${model} in ${datasets}_egs with ${loss} with mean normalization \033[0m\n"
    python TrainAndTest/Spectrogram/train_egs.py \
      --model ${model} \
      --train-dir ${lstm_dir}/data/${datasets}/egs/spect/dev_log \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/spect/valid_log \
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
      --channels 16,64,128,256 \
      --batch-size 256 \
      --embedding-size 128 \
      --avg-size 4 \
      --num-center 2 \
      --num-valid 2 \
      --alpha 12 \
      --margin 0.3 \
      --grad-clip 0 \
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
