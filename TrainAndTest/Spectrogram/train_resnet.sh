#!/usr/bin/env bash

stage=20

waited=0
while [ $(ps 17809 | wc -l) -eq 2 ]; do
  sleep 60
  waited=$(expr $waited + 1)
  echo -en "\033[1;4;31m Having waited for ${waited} minutes!\033[0m\r"
done
lstm_dir=/home/work2020/yangwenhao/project/lstm_speaker_verification

if [ $stage -le 0 ]; then
  for loss in soft asoft amsoft center; do
    echo -e "\n\033[1;4;31m Training with ${loss}\033[0m\n"
    python TrainAndTest/Spectrogram/train_resnet.py \
      --model ResNet \
      --resnet-size 18 \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/dev_257 \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_spect/test_257 \
      --embedding-size 512 \
      --batch-size 128 \
      --test-batch-size 32 \
      --nj 12 \
      --epochs 24 \
      --milestones 10,15,20 \
      --lr 0.1 \
      --margin 0.35 \
      --s 30 \
      --m 4 \
      --veri-pairs 12800 \
      --check-path Data/checkpoint/ResNet/18/spect/${loss} \
      --resume Data/checkpoint/ResNet/18/spect/${loss}/checkpoint_1.pth \
      --loss-type ${loss}
  done
fi

if [ $stage -le 20 ]; then
  datasets=vox1
  model=ThinResNet
  resnet_size=34
  encoder_type=STAP
  alpha=0
  block_type=None
  embedding_size=128
  input_norm=Mean
  loss=soft

  for loss in soft; do
    echo -e "\n\033[1;4;31m Training ${model}${resnet_size} in ${datasets}_egs with ${loss} \033[0m\n"
    python TrainAndTest/Spectrogram/train_egs.py \
      --model ${model} \
      --train-dir ${lstm_dir}/data/${datasets}/egs/spect/dev_log \
      --train-test-dir ${lstm_dir}/data/vox1/spect/dev_log/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/spect/valid_log \
      --test-dir ${lstm_dir}/data/vox1/spect/test_log \
      --feat-format kaldi \
      --fix-length \
      --input-norm ${input_norm} \
      --resnet-size ${resnet_size} \
      --nj 12 \
      --epochs 50 \
      --accu-steps 1 \
      --lr 0.1 \
      --milestones 21,41,48 \
      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/spect_egs_mean/${loss}/${encoder_type}_em${embedding_size}_alpha${alpha}_wde4 \
      --resume Data/checkpoint/${model}${resnet_size}/${datasets}/spect_egs_mean/${loss}/${encoder_type}_em${embedding_size}_alpha${alpha}_wde4/checkpoint_10.pth \
      --channels 16,32,64,128 \
      --input-dim 161 \
      --block-type ${block_type} \
      --stride 2 \
      --batch-size 128 \
      --embedding-size ${embedding_size} \
      --time-dim 1 \
      --avg-size 4 \
      --encoder-type ${encoder_type} \
      --num-valid 2 \
      --alpha ${alpha} \
      --margin 0.2 \
      --grad-clip 0 \
      --s 30 \
      --lr-ratio 0.01 \
      --weight-decay 0.0001 \
      --dropout-p 0 \
      --gpu-id 0,1 \
      --all-iteraion 0 \
      --extract \
      --cos-sim \
      --loss-type ${loss}
  done
  exit
fi
