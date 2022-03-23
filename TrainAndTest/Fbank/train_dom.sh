#!/usr/bin/env bash

stage=70

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
      --random-chunk 300 400 \
      --nj 10 \
      --epochs 60 \
      --lr 0.1 \
      --milestones 12,24,36,48 \
      --input-dim ${input_dim} \
      --channels 512,512,512,512,1500 \
      --encoder-type ${encod} \
      --check-path Data/checkpoint/${model}/${datasets}/${feat_type}_egs_revg/${loss}/feat${feat}_input${input_norm}_${encod}_em${embedding_size}_wde3_step5_domain2dr1_longer \
      --resume Data/checkpoint/${model}/${datasets}/${feat_type}_egs_revg/${loss}/feat${feat}_input${input_norm}_${encod}_em${embedding_size}_wde3_step5_domain2dr1_longer/checkpoint_21.pth \
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

if [ $stage -le 60 ]; then
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
    python TrainAndTest/train_egs_binary.py \
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
      --check-path Data/checkpoint/${model}/${datasets}/${feat_type}_egs_binary/${loss}/feat${feat}_input${input_norm}_${encod}_em${embedding_size}_wde3_step5_var \
      --resume Data/checkpoint/${model}/${datasets}/${feat_type}_egs_binary/${loss}/feat${feat}_input${input_norm}_${encod}_em${embedding_size}_wde3_step5_var/checkpoint_21.pth \
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

if [ $stage -le 70 ]; then
#  lstm_dir=/home/work2020/yangwenhao/project/lstm_speaker_verification
  datasets=cnceleb
  testset=cnceleb
  feat_type=klfb
  model=ThinResNet
  resnet_size=34
  encoder_type=SAP2
  embedding_size=512
  block_type=basic
  downsample=k3
  kernel=5,5
  loss=arcsoft
  alpha=0
  input_norm=Mean
  mask_layer=baseline
  scheduler=rop
  optimizer=sgd
  input_dim=40
  batch_size=256
  fast=none1
  mask_layer=advbinary
  weight=vox2_rcf
  scale=0.2
  subset=
  stat_type=maxmargin
        # --milestones 15,25,35,45 \

  for loss in arcsoft; do
    echo -e "\n\033[1;4;31m Stage${stage}: Training ${model}${resnet_size} in ${datasets}_egs with ${loss} with ${input_norm} normalization \033[0m\n"
     python TrainAndTest/train_egs_binary.py \
       --model ${model} \
       --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev${subset}_fb${input_dim} \
       --train-test-dir ${lstm_dir}/data/${testset}/${feat_type}/dev_fb${input_dim}/trials_dir \
       --train-trials trials_2w \
       --shuffle \
       --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev${subset}_fb${input_dim}_valid \
       --test-dir ${lstm_dir}/data/${testset}/${feat_type}/test_fb${input_dim} \
       --feat-format kaldi \
       --random-chunk 200 400 \
       --input-norm ${input_norm} \
       --resnet-size ${resnet_size} \
       --nj 12 \
       --epochs 60 \
       --batch-size ${batch_size} \
       --optimizer ${optimizer} \
       --scheduler ${scheduler} \
       --lr 0.1 \
       --base-lr 0.000006 \
       --mask-layer ${mask_layer} \
       --milestones 10,20,30,40,50 \
       --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}${input_dim}_egs${subset}_${mask_layer}/${loss}_${optimizer}_${scheduler}/${input_norm}_batch${batch_size}_${block_type}_down${downsample}_none1_${encoder_type}_dp01_alpha${alpha}_em${embedding_size}_${stat_type}_wd5e4_var \
       --resume Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}${input_dim}_egs${subset}_${mask_layer}/${loss}_${optimizer}_${scheduler}/${input_norm}_batch${batch_size}_${block_type}_down${downsample}_none1_${encoder_type}_dp01_alpha${alpha}_em${embedding_size}_${stat_type}_wd5e4_var/checkpoint_60.pth \
       --kernel-size ${kernel} \
       --downsample ${downsample} \
       --channels 16,32,64,128 \
       --fast none1 \
       --stride 2,1 \
       --block-type ${block_type} \
       --embedding-size ${embedding_size} \
       --time-dim 1 \
       --avg-size 5 \
       --encoder-type ${encoder_type} \
       --num-valid 2 \
       --alpha ${alpha} \
       --margin 0.2 \
       --s 30 \
       --weight-decay 0.0005 \
       --dropout-p 0.1 \
       --gpu-id 0,1 \
       --extract \
       --domain \
       --cos-sim \
       --all-iteraion 0 \
       --remove-vad \
       --submean \
       --loss-type ${loss}

#    python TrainAndTest/train_egs.py \
#      --model ${model} \
#      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev${subset}_fb${input_dim} \
#      --train-test-dir ${lstm_dir}/data/${datasets}/${feat_type}/dev_fb${input_dim}/trials_dir \
#      --train-trials trials_2w \
#      --shuffle \
#      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev${subset}_fb${input_dim}_valid \
#      --test-dir ${lstm_dir}/data/${testset}/${feat_type}/test_fb${input_dim} \
#      --feat-format kaldi \
#      --random-chunk 200 400 \
#      --input-norm ${input_norm} \
#      --resnet-size ${resnet_size} \
#      --nj 12 \
#      --epochs 60 \
#      --batch-size ${batch_size} \
#      --optimizer ${optimizer} \
#      --scheduler ${scheduler} \
#      --lr 0.001 \
#      --base-lr 0.00000001 \
#      --mask-layer ${mask_layer} \
#      --init-weight ${weight} \
#      --scale ${scale} \
#      --milestones 10,20,30,40,50 \
#      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs${subset}_${mask_layer}/${loss}_${optimizer}_${scheduler}/${input_norm}_batch${batch_size}_${block_type}_down${downsample}_${fast}_${encoder_type}_dp01_alpha${alpha}_em${embedding_size}_wd5e4_var \
#      --resume Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs${subset}_${mask_layer}/${loss}_${optimizer}_${scheduler}/${input_norm}_batch${batch_size}_${block_type}_down${downsample}_${fast}_${encoder_type}_dp01_alpha${alpha}_em${embedding_size}_wd5e4_var/checkpoint_60.pth \
#      --kernel-size ${kernel} \
#      --downsample ${downsample} \
#      --channels 16,32,64,128 \
#      --fast ${fast} \
#      --stride 2,1 \
#      --block-type ${block_type} \
#      --embedding-size ${embedding_size} \
#      --time-dim 1 \
#      --avg-size 5 \
#      --encoder-type ${encoder_type} \
#      --num-valid 2 \
#      --alpha ${alpha} \
#      --margin 0.2 \
#      --s 30 \
#      --weight-decay 0.0005 \
#      --dropout-p 0.1 \
#      --gpu-id 0,1 \
#      --extract \
#      --cos-sim \
#      --all-iteraion 0 \
#      --remove-vad \
#      --loss-type ${loss}
  done
#exit
fi

if [ $stage -le 70 ]; then
#  lstm_dir=/home/work2020/yangwenhao/project/lstm_speaker_verification
  datasets=cnceleb
  testset=cnceleb
  feat_type=klfb
  model=ThinResNet
  resnet_size=34
  encoder_type=SAP2
  embedding_size=512
  block_type=basic
  downsample=k3
  kernel=5,5
  loss=arcsoft
  alpha=0
  input_norm=Mean
  mask_layer=baseline
  scheduler=rop
  optimizer=sgd
  input_dim=40
  batch_size=256
  fast=none1
  mask_layer=advbiframe
  weight=vox2_rcf
  scale=0.2
  subset=
  stat_type=maxmargin
        # --milestones 15,25,35,45 \

  for loss in arcsoft; do
    echo -e "\n\033[1;4;31m Stage${stage}: Training ${model}${resnet_size} in ${datasets}_egs with ${loss} with ${input_norm} normalization \033[0m\n"
     python TrainAndTest/train_egs_binary_frame.py \
       --model ${model} \
       --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev${subset}_fb${input_dim} \
       --train-test-dir ${lstm_dir}/data/${testset}/${feat_type}/dev_fb${input_dim}/trials_dir \
       --train-trials trials_2w \
       --shuffle \
       --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev${subset}_fb${input_dim}_valid \
       --test-dir ${lstm_dir}/data/${testset}/${feat_type}/test_fb${input_dim} \
       --feat-format kaldi \
       --random-chunk 200 400 \
       --input-norm ${input_norm} \
       --resnet-size ${resnet_size} \
       --nj 12 \
       --epochs 60 \
       --batch-size ${batch_size} \
       --optimizer ${optimizer} \
       --scheduler ${scheduler} \
       --lr 0.1 \
       --base-lr 0.000006 \
       --mask-layer ${mask_layer} \
       --milestones 10,20,30,40,50 \
       --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}${input_dim}_egs${subset}_${mask_layer}/${loss}_${optimizer}_${scheduler}/${input_norm}_batch${batch_size}_${block_type}_down${downsample}_none1_${encoder_type}_dp01_alpha${alpha}_em${embedding_size}_${stat_type}_wd5e4_var \
       --resume Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}${input_dim}_egs${subset}_${mask_layer}/${loss}_${optimizer}_${scheduler}/${input_norm}_batch${batch_size}_${block_type}_down${downsample}_none1_${encoder_type}_dp01_alpha${alpha}_em${embedding_size}_${stat_type}_wd5e4_var/checkpoint_60.pth \
       --kernel-size ${kernel} \
       --downsample ${downsample} \
       --channels 16,32,64,128 \
       --fast none1 \
       --stride 2,1 \
       --block-type ${block_type} \
       --embedding-size ${embedding_size} \
       --time-dim 1 \
       --avg-size 5 \
       --encoder-type ${encoder_type} \
       --num-valid 2 \
       --alpha ${alpha} \
       --margin 0.2 \
       --s 30 \
       --weight-decay 0.0005 \
       --dropout-p 0.1 \
       --gpu-id 0,1 \
       --extract \
       --domain \
       --cos-sim \
       --all-iteraion 0 \
       --remove-vad \
       --submean \
       --loss-type ${loss}

#    python TrainAndTest/train_egs.py \
#      --model ${model} \
#      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev${subset}_fb${input_dim} \
#      --train-test-dir ${lstm_dir}/data/${datasets}/${feat_type}/dev_fb${input_dim}/trials_dir \
#      --train-trials trials_2w \
#      --shuffle \
#      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev${subset}_fb${input_dim}_valid \
#      --test-dir ${lstm_dir}/data/${testset}/${feat_type}/test_fb${input_dim} \
#      --feat-format kaldi \
#      --random-chunk 200 400 \
#      --input-norm ${input_norm} \
#      --resnet-size ${resnet_size} \
#      --nj 12 \
#      --epochs 60 \
#      --batch-size ${batch_size} \
#      --optimizer ${optimizer} \
#      --scheduler ${scheduler} \
#      --lr 0.001 \
#      --base-lr 0.00000001 \
#      --mask-layer ${mask_layer} \
#      --init-weight ${weight} \
#      --scale ${scale} \
#      --milestones 10,20,30,40,50 \
#      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs${subset}_${mask_layer}/${loss}_${optimizer}_${scheduler}/${input_norm}_batch${batch_size}_${block_type}_down${downsample}_${fast}_${encoder_type}_dp01_alpha${alpha}_em${embedding_size}_wd5e4_var \
#      --resume Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs${subset}_${mask_layer}/${loss}_${optimizer}_${scheduler}/${input_norm}_batch${batch_size}_${block_type}_down${downsample}_${fast}_${encoder_type}_dp01_alpha${alpha}_em${embedding_size}_wd5e4_var/checkpoint_60.pth \
#      --kernel-size ${kernel} \
#      --downsample ${downsample} \
#      --channels 16,32,64,128 \
#      --fast ${fast} \
#      --stride 2,1 \
#      --block-type ${block_type} \
#      --embedding-size ${embedding_size} \
#      --time-dim 1 \
#      --avg-size 5 \
#      --encoder-type ${encoder_type} \
#      --num-valid 2 \
#      --alpha ${alpha} \
#      --margin 0.2 \
#      --s 30 \
#      --weight-decay 0.0005 \
#      --dropout-p 0.1 \
#      --gpu-id 0,1 \
#      --extract \
#      --cos-sim \
#      --all-iteraion 0 \
#      --remove-vad \
#      --loss-type ${loss}
  done
#exit
fi