#!/usr/bin/env bash

stage=0
waited=0
while [ `ps 75486 | wc -l` -eq 2 ]; do
  sleep 60
  waited=$(expr $waited + 1)
  echo -en "\033[1;4;31m Having waited for ${waited} minutes!\033[0m\r"
done
#stage=10


if [ $stage -le 0 ]; then
#  for loss in soft asoft ; do
  model=ExResNet
  datasets=vox1
  feat=fb64
  loss=soft
  for encod in None ; do
    echo -e "\n\033[1;4;31m Training ${model}_${encod} with ${loss}\033[0m\n"
    python -W ignore TrainAndTest/Fbank/ResNets/train_exres_kaldi.py \
      --train-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/vox1/pydb/dev_${feat} \
      --test-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/vox1/pydb/test_${feat} \
      --nj 15 \
      --epochs 30 \
      --milestones 12,19,25 \
      --model ${model} \
      --resnet-size 34 \
      --stride 2 \
      --feat-format kaldi \
      --embedding-size 128 \
      --batch-size 128 \
      --accu-steps 1 \
      --feat-dim 64 \
      --remove-vad \
      --time-dim 1 \
      --avg-size 4 \
      --kernel-size 5,5 \
      --test-input-per-file 4 \
      --lr 0.1 \
      --encoder-type ${encod} \
      --check-path Data/checkpoint/${model}34/${datasets}_${encod}/${feat}/${loss} \
      --resume Data/checkpoint/${model}34/${datasets}_${encod}/${feat}/${loss}/checkpoint_100.pth \
      --input-per-spks 384 \
      --veri-pairs 9600 \
      --gpu-id 0 \
      --num-valid 2 \
      --loss-type soft
  done

  for encod in STAP ; do
    echo -e "\n\033[1;4;31m Training ${model}_${encod} with ${loss}\033[0m\n"
    python -W ignore TrainAndTest/Fbank/ResNets/train_exres_kaldi.py \
      --train-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/vox1/pydb/dev_${feat} \
      --test-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/vox1/pydb/test_${feat} \
      --nj 15 \
      --epochs 30 \
      --milestones 12,19,25 \
      --model ${model} \
      --resnet-size 34 \
      --stride 2 \
      --feat-format kaldi \
      --embedding-size 128 \
      --batch-size 128 \
      --accu-steps 1 \
      --feat-dim 64 \
      --remove-vad \
      --time-dim 8 \
      --avg-size 1 \
      --kernel-size 5,5 \
      --test-input-per-file 4 \
      --lr 0.1 \
      --encoder-type ${encod} \
      --check-path Data/checkpoint/${model}34/${datasets}_${encod}/${feat}/${loss} \
      --resume Data/checkpoint/${model}34/${datasets}_${encod}/${feat}/${loss}/checkpoint_100.pth \
      --input-per-spks 384 \
      --veri-pairs 9600 \
      --gpu-id 0 \
      --num-valid 2 \
      --loss-type soft
  done
fi
stage=100

if [ $stage -le 1 ]; then
#  for loss in center amsoft ; do/
  for loss in center asoft; do
    echo -e "\n\033[1;4;31m Finetuning ${model} with ${loss}\033[0m\n"
    python -W ignore TrainAndTest/Fbank/ResNets/train_exres_kaldi.py \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb64/dev_noc \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb64/test_noc \
      --nj 12 \
      --model ExResNet34 \
      --resnet-size 34 \
      --feat-dim 64 \
      --stride 1 \
      --kernel-size 3,3 \
      --batch-size 64 \
      --check-path Data/checkpoint/${model}/spect/${loss} \
      --resume Data/checkpoint/${model}/spect/soft/checkpoint_30.pth \
      --input-per-spks 192 \
      --loss-type ${loss} \
      --lr 0.01 \
      --loss-ratio 0.01 \
      --milestones 5,9 \
      --num-valid 2 \
      --epochs 12
  done

fi

if [ $stage -le 10 ]; then
#  for loss in soft asoft ; do
  model=ExResNet34
  datasets=vox1
  feat=fb64_wcmvn
  for loss in soft ; do
    echo -e "\n\033[1;4;31m Training ${model} with ${loss}\033[0m\n"
    python -W ignore TrainAndTest/Fbank/ResNets/train_exres_kaldi.py \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb/dev_fb64_wcmvn \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb/test_fb64_wcmvn \
      --nj 14 \
      --epochs 20 \
      --milestones 8,12,16 \
      --model ${model} \
      --resnet-size 34 \
      --embedding-size 128 \
      --feat-dim 64 \
      --remove-vad \
      --stride 1 \
      --time-dim 1 \
      --avg-size 1 \
      --kernel-size 3,3 \
      --batch-size 64 \
      --test-batch-size 32 \
      --test-input-per-file 2 \
      --lr 0.1 \
      --check-path Data/checkpoint/${model}/${datasets}/${feat}/${loss}_fix \
      --resume Data/checkpoint/${model}/${datasets}/${feat}/${loss}_fix/checkpoint_1.pth \
      --input-per-spks 192 \
      --veri-pairs 9600 \
      --gpu-id 1 \
      --num-valid 2 \
      --loss-type ${loss}
  done
fi

if [ $stage -le 15 ]; then
#  for loss in soft asoft ; do
  model=SiResNet34
  datasets=vox1
  feat=fb64_cmvn
  for loss in soft ; do
    echo -e "\n\033[1;4;31m Training ${model} with ${loss}\033[0m\n"
    python -W ignore TrainAndTest/Fbank/ResNets/train_exres_kaldi.py \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb/dev_fb64 \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb/test_fb64 \
      --nj 16 \
      --epochs 13 \
      --milestones 1,5,10 \
      --model ${model} \
      --resnet-size 34 \
      --embedding-size 128 \
      --feat-dim 64 \
      --remove-vad \
      --stride 1 \
      --time-dim 1 \
      --avg-size 1 \
      --kernel-size 3,3 \
      --batch-size 64 \
      --test-batch-size 4 \
      --test-input-per-file 4 \
      --lr 0.1 \
      --check-path Data/checkpoint/${model}/${datasets}/${feat}/${loss} \
      --resume Data/checkpoint/${model}/${datasets}/${feat}/${loss}/checkpoint_9.pth \
      --input-per-spks 192 \
      --veri-pairs 9600 \
      --gpu-id 0 \
      --num-valid 2 \
      --loss-type ${loss}
  done
fi