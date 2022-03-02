#!/usr/bin/env bash

stage=100
waited=0
while [ `ps 1141965 | wc -l` -eq 2 ]; do
  sleep 60
  waited=$(expr $waited + 1)
  echo -en "\033[1;4;31m Having waited for ${waited} minutes!\033[0m\r"
done
#stage=10

lstm_dir=/home/yangwenhao/project/lstm_speaker_verification

if [ $stage -le 0 ]; then
#  for loss in soft asoft ; do
  model=ExResNet
  datasets=vox1
  feat=power_257
  loss=soft
#  for encod in None ; do
#    echo -e "\n\033[1;4;31m Training ${model}_${encod} with ${loss}\033[0m\n"
#    python -W ignore TrainAndTest/Fbank/ResNets/train_exres_kaldi.py \
#      --train-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/vox1/pydb/dev_${feat} \
#      --test-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/vox1/pydb/test_${feat} \
#      --nj 10 \
#      --epochs 30 \
#      --milestones 12,19,25 \
#      --model ${model} \
#      --resnet-size 34 \
#      --stride 2 \
#      --feat-format kaldi \
#      --embedding-size 128 \
#      --batch-size 128 \
#      --accu-steps 1 \
#      --feat-dim 64 \
#      --remove-vad \
#      --time-dim 1 \
#      --avg-size 4 \
#      --kernel-size 5,5 \
#      --test-input-per-file 4 \
#      --lr 0.1 \
#      --encoder-type ${encod} \
#      --check-path Data/checkpoint/${model}34/${datasets}_${encod}/${feat}/${loss} \
#      --resume Data/checkpoint/${model}34/${datasets}_${encod}/${feat}/${loss}/checkpoint_100.pth \
#      --input-per-spks 384 \
#      --veri-pairs 9600 \
#      --gpu-id 0 \
#      --num-valid 2 \
#      --loss-type soft
#  done

  for encod in STAP ; do
    echo -e "\n\033[1;4;31m Training ${model}_${encod} with ${loss}\033[0m\n"
    python -W ignore TrainAndTest/Fbank/ResNets/train_exres_kaldi.py \
      --train-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/vox1/spect/dev_${feat} \
      --test-dir /home/work2020/yangwenhao/project/lstm_speaker_verification/data/vox1/spect/test_${feat} \
      --nj 12 \
      --epochs 25 \
      --milestones 10,15,20 \
      --model ${model} \
      --resnet-size 34 \
      --stride 2 \
      --inst-norm \
      --filter \
      --feat-format kaldi \
      --embedding-size 128 \
      --batch-size 128 \
      --accu-steps 1 \
      --feat-dim 64 \
      --time-dim 8 \
      --avg-size 1 \
      --kernel-size 5,5 \
      --test-input-per-file 4 \
      --lr 0.1 \
      --loss-ratio 0.1 \
      --encoder-type ${encod} \
      --check-path Data/checkpoint/${model}34_filter/${datasets}_${encod}/${feat}/${loss}_mean \
      --resume Data/checkpoint/${model}34_filter/${datasets}_${encod}/${feat}/${loss}_mean/checkpoint_100.pth \
      --input-per-spks 384 \
      --veri-pairs 9600 \
      --gpu-id 0 \
      --num-valid 2 \
      --loss-type soft

  done
fi
#stage=100

if [ $stage -le 1 ]; then
#  for loss in center amsoft ; do/
  for loss in center asoft; do
    echo -e "\n\033[1;4;31m Finetuning ${model} with ${loss}\033[0m\n"
    python -W ignore TrainAndTest/Fbank/ResNets/train_exres_kaldi.py \
      --train-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb64/dev_noc \
      --test-dir /home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb64/test_noc \
      --nj 4 \
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

if [ $stage -le 20 ]; then
  lstm_dir=/home/work2020/yangwenhao/project/lstm_speaker_verification
  model=ExResNet
  datasets=vox1
  feat=power_257
  loss=soft

  for encod in STAP ; do
    echo -e "\n\033[1;4;31m Training ${model}_${encod} with ${loss}\033[0m\n"
    python -W ignore TrainAndTest/Spectrogram/train_egs.py \
      --train-dir ${lstm_dir}/data/vox1/egs/spect/dev_${feat} \
      --valid-dir ${lstm_dir}/data/vox1/egs/spect/valid_${feat} \
      --test-dir ${lstm_dir}/data/vox1/spect/test_${feat} \
      --nj 10 \
      --epochs 22 \
      --milestones 8,13,18 \
      --model ${model} \
      --resnet-size 34 \
      --stride 2 \
      --inst-norm \
      --filter \
      --feat-format kaldi \
      --embedding-size 128 \
      --batch-size 128 \
      --accu-steps 1 \
      --feat-dim 64 \
      --input-dim 257 \
      --time-dim 8 \
      --avg-size 1 \
      --kernel-size 5,5 \
      --test-input-per-file 4 \
      --lr 0.1 \
      --loss-ratio 0.1 \
      --encoder-type ${encod} \
      --check-path Data/checkpoint/${model}34_filter/${datasets}_${encod}/${feat}/${loss}_mean_0.5_0.05 \
      --resume Data/checkpoint/${model}34_filter/${datasets}_${encod}/${feat}/${loss}_mean_0.5_0.05/checkpoint_100.pth \
      --input-per-spks 384 \
      --cos-sim \
      --veri-pairs 9600 \
      --gpu-id 0 \
      --num-valid 2 \
      --loss-type soft

  done
fi

if [ $stage -le 40 ]; then
  lstm_dir=/home/yangwenhao/project/lstm_speaker_verification
  datasets=vox1
  testset=vox1
  feat_type=klfb
  model=ThinResNet
  resnet_size=18
  encoder_type=SAP2
  embedding_size=256
  block_type=cbam
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

#  loss=soft
  encoder_type=SAP2
#  for input_dim in 24; do
#    echo -e "\n\033[1;4;31m Stage${stage}: Training ${model}${resnet_size} in ${datasets}_egs with ${loss} with ${input_norm} normalization \033[0m\n"
#    python TrainAndTest/train_egs.py \
#      --model ${model} \
#      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev_fb${input_dim} \
#      --train-test-dir ${lstm_dir}/data/${testset}/${feat_type}/dev_fb${input_dim}/trials_dir \
#      --train-trials trials_2w \
#      --shuffle \
#      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev_fb${input_dim}_valid \
#      --test-dir ${lstm_dir}/data/${testset}/${feat_type}/test_fb${input_dim} \
#      --feat-format kaldi \
#      --random-chunk 200 400 \
#      --input-norm ${input_norm} \
#      --resnet-size ${resnet_size} \
#      --nj 12 \
#      --epochs 50 \
#      --batch-size ${batch_size} \
#      --optimizer ${optimizer} \
#      --scheduler ${scheduler} \
#      --lr 0.1 \
#      --base-lr 0.000006 \
#      --mask-layer ${mask_layer} \
#      --milestones 10,20,30,40 \
#      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}${input_dim}_egs_${mask_layer}/${loss}_${optimizer}_${scheduler}/${input_norm}_batch${batch_size}_${block_type}_down${downsample}_${fast}_${encoder_type}_dp01_alpha${alpha}_em${embedding_size}_wd5e4_var \
#      --resume Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}${input_dim}_egs_${mask_layer}/${loss}_${optimizer}_${scheduler}/${input_norm}_batch${batch_size}_${block_type}_down${downsample}_${fast}_${encoder_type}_dp01_alpha${alpha}_em${embedding_size}_wd5e4_var/checkpoint_50.pth \
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
#      --loss-type ${loss} \
#      --margin 0.2 \
#      --s 30 \
#      --weight-decay 0.0005 \
#      --dropout-p 0.1 \
#      --gpu-id 0,1 \
#      --extract \
#      --cos-sim \
#      --all-iteraion 0 \
#      --remove-vad
#  done

  for input_dim in 64 80 ; do
    echo -e "\n\033[1;4;31m Stage${stage}: Training ${model}${resnet_size} in ${datasets}_egs with ${loss} with ${input_norm} normalization \033[0m\n"
    python TrainAndTest/train_egs.py \
      --model ${model} \
      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev_fb${input_dim} \
      --train-test-dir ${lstm_dir}/data/${testset}/${feat_type}/dev_fb${input_dim}/trials_dir \
      --train-trials trials_2w \
      --shuffle \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev_fb${input_dim}_valid \
      --test-dir ${lstm_dir}/data/${testset}/${feat_type}/test_fb${input_dim} \
      --feat-format kaldi \
      --random-chunk 200 400 \
      --input-norm ${input_norm} \
      --resnet-size ${resnet_size} \
      --nj 8 \
      --epochs 50 \
      --batch-size ${batch_size} \
      --optimizer ${optimizer} \
      --scheduler ${scheduler} \
      --lr 0.1 \
      --base-lr 0.000006 \
      --mask-layer ${mask_layer} \
      --milestones 10,20,30,40 \
      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}${input_dim}_egs_${mask_layer}/${loss}_${optimizer}_${scheduler}/${input_norm}_batch${batch_size}_${block_type}_down${downsample}_${fast}_${encoder_type}_dp01_alpha${alpha}_em${embedding_size}_wd5e4_var \
      --resume Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}${input_dim}_egs_${mask_layer}/${loss}_${optimizer}_${scheduler}/${input_norm}_batch${batch_size}_${block_type}_down${downsample}_${fast}_${encoder_type}_dp01_alpha${alpha}_em${embedding_size}_wd5e4_var/checkpoint_50.pth \
      --kernel-size ${kernel} \
      --downsample ${downsample} \
      --channels 16,32,64,128 \
      --fast ${fast} \
      --stride 2,1 \
      --block-type ${block_type} \
      --embedding-size ${embedding_size} \
      --time-dim 1 \
      --avg-size 5 \
      --encoder-type ${encoder_type} \
      --num-valid 2 \
      --alpha ${alpha} \
      --loss-type ${loss} \
      --margin 0.2 \
      --s 30 \
      --weight-decay 0.0005 \
      --dropout-p 0.1 \
      --gpu-id 2,3 \
      --extract \
      --cos-sim \
      --all-iteraion 0 \
      --remove-vad
  done

#  for input_dim in 64 80; do
#    echo -e "\n\033[1;4;31m Stage${stage}: Training ${model}${resnet_size} in ${datasets}_egs with ${loss} with ${input_norm} normalization \033[0m\n"
#    python TrainAndTest/train_egs.py \
#      --model ${model} \
#      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev_fb${input_dim} \
#      --train-test-dir ${lstm_dir}/data/${testset}/${feat_type}/dev_fb${input_dim}/trials_dir \
#      --train-trials trials_2w \
#      --shuffle \
#      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev_fb${input_dim}_valid \
#      --test-dir ${lstm_dir}/data/${testset}/${feat_type}/test_fb${input_dim} \
#      --feat-format kaldi \
#      --random-chunk 200 400 \
#      --input-norm ${input_norm} \
#      --resnet-size ${resnet_size} \
#      --nj 12 \
#      --epochs 50 \
#      --batch-size ${batch_size} \
#      --optimizer ${optimizer} \
#      --scheduler ${scheduler} \
#      --lr 0.1 \
#      --base-lr 0.000006 \
#      --mask-layer ${mask_layer} \
#      --milestones 10,20,30,40 \
#      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}${input_dim}_egs_${mask_layer}/${loss}_${optimizer}_${scheduler}/${input_norm}_batch${batch_size}_${block_type}_down${downsample}_${fast}_${encoder_type}_dp01_alpha${alpha}_em${embedding_size}_wd5e4_var \
#      --resume Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}${input_dim}_egs_${mask_layer}/${loss}_${optimizer}_${scheduler}/${input_norm}_batch${batch_size}_${block_type}_down${downsample}_${fast}_${encoder_type}_dp01_alpha${alpha}_em${embedding_size}_wd5e4_var/checkpoint_50.pth \
#      --kernel-size ${kernel} \
#      --downsample ${downsample} \
#      --channels 16,32,64,128 \
#      --fast ${fast} \
#      --stride 2,2 \
#      --block-type ${block_type} \
#      --embedding-size ${embedding_size} \
#      --time-dim 1 \
#      --avg-size 5 \
#      --encoder-type ${encoder_type} \
#      --num-valid 2 \
#      --alpha ${alpha} \
#      --loss-type ${loss} \
#      --margin 0.2 \
#      --s 30 \
#      --weight-decay 0.0005 \
#      --dropout-p 0.1 \
#      --gpu-id 0,1 \
#      --extract \
#      --cos-sim \
#      --all-iteraion 0 \
#      --remove-vad
#  done
  exit
fi


if [ $stage -le 41 ]; then
  lstm_dir=/home/work2020/yangwenhao/project/lstm_speaker_verification
  datasets=vox1
  feat_type=klfb
  model=ThinResNet
  resnet_size=34
  encoder_type=SAP2
  embedding_size=256
  block_type=basic
  downsample=None
  kernel=5,5
  loss=arcsoft
  alpha=0
  input_norm=Mean
  mask_layer=None
  scheduler=rop
  optimizer=sgd
  input_dim=40
  batch_size=256


  for power_weight in max ; do
    echo -e "\n\033[1;4;31m Stage${stage}: Training ${model}${resnet_size} in ${datasets}_egs with ${loss} with ${input_norm} normalization \033[0m\n"
    # python TrainAndTest/train_egs.py \
    #   --model ${model} \
    #   --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev_fb${input_dim} \
    #   --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/dev_fb${input_dim}/trials_dir \
    #   --train-trials trials_2w \
    #   --shuffle \
    #   --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/valid_fb${input_dim} \
    #   --test-dir ${lstm_dir}/data/vox1/${feat_type}/test_fb${input_dim} \
    #   --feat-format kaldi \
    #   --random-chunk 200 400 \
    #   --input-norm ${input_norm} \
    #   --resnet-size ${resnet_size} \
    #   --nj 12 \
    #   --epochs 50 \
    #   --batch-size ${batch_size} \
    #   --optimizer ${optimizer} \
    #   --scheduler ${scheduler} \
    #   --lr 0.1 \
    #   --base-lr 0.000006 \
    #   --mask-layer ${mask_layer} \
    #   --milestones 15,25,35,45 \
    #   --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_baseline/${loss}_${optimizer}_${scheduler}/${input_norm}_${block_type}_down${downsample}_none1_${encoder_type}_dp01_alpha${alpha}_em${embedding_size}_wd5e4_var \
    #   --resume Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_baseline/${loss}_${optimizer}_${scheduler}/${input_norm}_${block_type}_down${downsample}_none1_${encoder_type}_dp01_alpha${alpha}_em${embedding_size}_wd5e4_var/checkpoint_50.pth \
    #   --kernel-size ${kernel} \
    #   --downsample ${downsample} \
    #   --channels 16,32,64,128 \
    #   --fast none1 \
    #   --stride 2,1 \
    #   --block-type ${block_type} \
    #   --embedding-size ${embedding_size} \
    #   --time-dim 1 \
    #   --avg-size 5 \
    #   --encoder-type ${encoder_type} \
    #   --num-valid 2 \
    #   --alpha ${alpha} \
    #   --margin 0.2 \
    #   --s 30 \
    #   --weight-decay 0.0005 \
    #   --dropout-p 0.1 \
    #   --gpu-id 0,1 \
    #   --extract \
    #   --cos-sim \
    #   --all-iteraion 0 \
    #   --remove-vad \
    #   --loss-type ${loss}

    mask_layer=baseline
    weight=vox2_rcf
      #     --init-weight ${weight} \
      # --power-weight ${power_weight} \
      # _${weight}${power_weight}
    python TrainAndTest/train_egs.py \
      --model ${model} \
      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev_fb${input_dim} \
      --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/dev_fb${input_dim}/trials_dir \
      --train-trials trials_2w \
      --shuffle \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/valid_fb${input_dim} \
      --test-dir ${lstm_dir}/data/vox1/${feat_type}/test_fb${input_dim} \
      --feat-format kaldi \
      --random-chunk 200 400 \
      --input-norm ${input_norm} \
      --resnet-size ${resnet_size} \
      --nj 12 \
      --epochs 50 \
      --batch-size ${batch_size} \
      --optimizer ${optimizer} \
      --scheduler ${scheduler} \
      --lr 0.1 \
      --base-lr 0.000006 \
      --mask-layer ${mask_layer} \
      --milestones 15,25,35,45 \
      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_${mask_layer}/${loss}_${optimizer}_${scheduler}/${input_norm}_${block_type}_down${downsample}_none1_${encoder_type}_dp125_alpha${alpha}_em${embedding_size}_wd5e4_var \
      --resume Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_${mask_layer}/${loss}_${optimizer}_${scheduler}/${input_norm}_${block_type}_down${downsample}_none1_${encoder_type}_dp125_alpha${alpha}_em${embedding_size}_wd5e4_var/checkpoint_50.pth \
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
      --dropout-p 0.125 \
      --gpu-id 0,1 \
      --extract \
      --cos-sim \
      --all-iteraion 0 \
      --remove-vad \
      --loss-type ${loss}
  done
  exit
fi

if [ $stage -le 42 ]; then
  lstm_dir=/home/work2020/yangwenhao/project/lstm_speaker_verification
  datasets=vox2
  feat_type=klfb
  model=ThinResNet
  resnet_size=50
  encoder_type=SAP2
  embedding_size=256
  block_type=basic
  downsample=None
  kernel=5,5
  loss=arcsoft
  alpha=0
  input_norm=Mean
  mask_layer=None
  scheduler=rop
  optimizer=sgd
  input_dim=40

  for encoder_type in SAP2; do
    echo -e "\n\033[1;4;31m Stage${stage}: Training ${model}${resnet_size} in ${datasets}_egs with ${loss} with ${input_norm} normalization \033[0m\n"
    python TrainAndTest/train_egs.py \
      --model ${model} \
      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev_fb${input_dim} \
      --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/dev_fb${input_dim}/trials_dir \
      --train-trials trials_2w \
      --shuffle \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev_fb${input_dim}_valid \
      --test-dir ${lstm_dir}/data/vox1/${feat_type}/test_fb${input_dim} \
      --feat-format kaldi \
      --random-chunk 200 400 \
      --input-norm ${input_norm} \
      --resnet-size ${resnet_size} \
      --nj 12 \
      --epochs 60 \
      --batch-size 128 \
      --optimizer ${optimizer} \
      --scheduler ${scheduler} \
      --lr 0.1 \
      --base-lr 0.000006 \
      --mask-layer ${mask_layer} \
      --milestones 10,20,30,40,50 \
      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_baseline/${loss}_${optimizer}_${scheduler}/${input_norm}_${block_type}_down${downsample}_none1_${encoder_type}_dp01_alpha${alpha}_em${embedding_size}_wde4_var \
      --resume Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_baseline/${loss}_${optimizer}_${scheduler}/${input_norm}_${block_type}_down${downsample}_none1_${encoder_type}_dp01_alpha${alpha}_em${embedding_size}_wde4_var/checkpoint_50.pth \
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
      --weight-decay 0.0001 \
      --dropout-p 0.1 \
      --gpu-id 0,1 \
      --extract \
      --cos-sim \
      --all-iteraion 0 \
      --remove-vad \
      --loss-type ${loss}
  done
  exit
fi

if [ $stage -le 100 ]; then
#  lstm_dir=/home/work2020/yangwenhao/project/lstm_speaker_verification
  datasets=cnceleb
  testset=cnceleb
  feat_type=klfb
  model=ThinResNet
  resnet_size=34
  encoder_type=SAP2
  embedding_size=256
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
  mask_layer=baseline
  weight=vox2_rcf
  scale=0.2
  subset=
  stat_type=maxmargin
        # --milestones 15,25,35,45 \
#        _${stat_type}

  for loss in arcsoft; do
    echo -e "\n\033[1;4;31m Stage${stage}: Training ${model}${resnet_size} in ${datasets}_egs with ${loss} with ${input_norm} normalization \033[0m\n"
     python TrainAndTest/train_egs.py \
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
       --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}${input_dim}_egs${subset}_${mask_layer}/${loss}_${optimizer}_${scheduler}/${input_norm}_batch${batch_size}_${block_type}_down${downsample}_${fast}_${encoder_type}_dp20_alpha${alpha}_em${embedding_size}_chn32_wd5e4_var \
       --resume Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}${input_dim}_egs${subset}_${mask_layer}/${loss}_${optimizer}_${scheduler}/${input_norm}_batch${batch_size}_${block_type}_down${downsample}_${fast}_${encoder_type}_dp20_alpha${alpha}_em${embedding_size}_chn32_wd5e4_var/checkpoint_60.pth \
       --kernel-size ${kernel} \
       --downsample ${downsample} \
       --channels 32,64,128,256 \
       --fast ${fast} \
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
       --dropout-p 0.2 \
       --gpu-id 0,1 \
       --extract \
       --cos-sim \
       --all-iteraion 0 \
       --remove-vad \
       --stat-type ${stat_type} \
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


if [ $stage -le 101 ]; then
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
#  mask_layer=None
  scheduler=rop
  optimizer=sgd
  input_dim=40
  batch_size=256
  fast=none1
  mask_layer=baseline
  weight=vox2_rcf
  scale=0.2
  subset=
  stat_type=maxmargin
  loss_ratio=1
        # --milestones 15,25,35,45 \

  for loss in arcdist ; do
    echo -e "\n\033[1;4;31m Stage${stage}: Training ${model}${resnet_size} in ${datasets}_egs with ${loss} with ${input_norm} normalization \033[0m\n"
#     python TrainAndTest/train_egs.py \
#       --model ${model} \
#       --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev${subset}_fb${input_dim} \
#       --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/dev_fb${input_dim}/trials_dir \
#       --train-trials trials_2w \
#       --shuffle \
#       --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev${subset}_fb${input_dim}_valid \
#       --test-dir ${lstm_dir}/data/vox1/${feat_type}/test_fb${input_dim} \
#       --feat-format kaldi \
#       --random-chunk 200 400 \
#       --input-norm ${input_norm} \
#       --resnet-size ${resnet_size} \
#       --nj 12 \
#       --epochs 60 \
#       --batch-size ${batch_size} \
#       --optimizer ${optimizer} \
#       --scheduler ${scheduler} \
#       --lr 0.1 \
#       --base-lr 0.000006 \
#       --mask-layer ${mask_layer} \
#       --milestones 10,20,30,40,50 \
#       --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs${subset}_${mask_layer}/${loss}_${optimizer}_${scheduler}/${input_norm}_batch${batch_size}_${block_type}_down${downsample}_none1_${encoder_type}_dp1584_alpha${alpha}_em${embedding_size}_wd5e4_var \
#       --resume Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs${subset}_${mask_layer}/${loss}_${optimizer}_${scheduler}/${input_norm}_batch${batch_size}_${block_type}_down${downsample}_none1_${encoder_type}_dp1584_alpha${alpha}_em${embedding_size}_wd5e4_var/checkpoint_60.pth \
#       --kernel-size ${kernel} \
#       --downsample ${downsample} \
#       --channels 16,32,64,128 \
#       --fast none1 \
#       --stride 2,1 \
#       --block-type ${block_type} \
#       --embedding-size ${embedding_size} \
#       --time-dim 1 \
#       --avg-size 5 \
#       --encoder-type ${encoder_type} \
#       --num-valid 2 \
#       --alpha ${alpha} \
#       --margin 0.2 \
#       --s 30 \
#       --weight-decay 0.0005 \
#       --dropout-p 0.1584 \
#       --gpu-id 0,1 \
#       --extract \
#       --cos-sim \
#       --all-iteraion 0 \
#       --remove-vad \
#       --loss-type ${loss}

#_lr${stat_type}${loss_ratio}
#_lr${stat_type}${loss_ratio}
#${input_dim}
    python TrainAndTest/train_egs.py \
      --model ${model} \
      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev${subset}_fb${input_dim} \
      --train-test-dir ${lstm_dir}/data/${datasets}/${feat_type}/dev_fb${input_dim}/trials_dir \
      --train-trials trials_2w \
      --shuffle \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev${subset}_fb${input_dim}_valid \
      --test-dir ${lstm_dir}/data/${testset}/${feat_type}/test_fb${input_dim} \
      --feat-format kaldi \
      --random-chunk 200 400 \
      --input-norm ${input_norm} \
      --resnet-size ${resnet_size} \
      --nj 6 \
      --epochs 60 \
      --batch-size ${batch_size} \
      --optimizer ${optimizer} \
      --scheduler ${scheduler} \
      --lr 0.1 \
      --base-lr 0.00000001 \
      --mask-layer ${mask_layer} \
      --init-weight ${weight} \
      --scale ${scale} \
      --milestones 10,20,30,40,50 \
      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs${subset}_${mask_layer}/${loss}_lncl_${optimizer}_${scheduler}/${input_norm}_batch${batch_size}_${block_type}_down${downsample}_${fast}_${encoder_type}_dp01_alpha${alpha}_em${embedding_size}_wd5e4_var \
      --resume Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs${subset}_${mask_layer}/${loss}_lncl_${optimizer}_${scheduler}/${input_norm}_batch${batch_size}_${block_type}_down${downsample}_${fast}_${encoder_type}_dp01_alpha${alpha}_em${embedding_size}_wd5e4_var/checkpoint_60.pth \
      --kernel-size ${kernel} \
      --downsample ${downsample} \
      --channels 16,32,64,128 \
      --fast ${fast} \
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
      --num-center 3 \
      --weight-decay 0.0005 \
      --dropout-p 0.1 \
      --gpu-id 0,1 \
      --extract \
      --cos-sim \
      --all-iteraion 0 \
      --remove-vad \
      --loss-type ${loss} \
      --lncl \
      --stat-type ${stat_type} \
      --loss-ratio ${loss_ratio}
  done
  exit
fi


if [ $stage -le 102 ]; then
  lstm_dir=/home/work2020/yangwenhao/project/lstm_speaker_verification
  datasets=cnceleb
  testset=cnceleb
  feat_type=klfb
  model=ThinResNet
  resnet_size=18
  encoder_type=SAP2
  embedding_size=256
  block_type=basic
  downsample=k3
  kernel=5,5
  loss=arcsoft
  alpha=0
  input_norm=Mean
  mask_layer=None
  scheduler=rop
  optimizer=sgd
  input_dim=40
  batch_size=256
  fast=none1
  mask_layer=baseline
  weight=one
  scale=0.2
  weight_p=0.1584
  subset=
        # --milestones 15,25,35,45 \

  for mask_layer in drop ; do
    echo -e "\n\033[1;4;31m Stage${stage}: Training ${model}${resnet_size} in ${datasets}_egs with ${loss} with ${input_norm} normalization \033[0m\n"
#     python TrainAndTest/train_egs.py \
#       --model ${model} \
#       --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev${subset}_fb${input_dim} \
#       --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/dev_fb${input_dim}/trials_dir \
#       --train-trials trials_2w \
#       --shuffle \
#       --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev${subset}_fb${input_dim}_valid \
#       --test-dir ${lstm_dir}/data/vox1/${feat_type}/test_fb${input_dim} \
#       --feat-format kaldi \
#       --random-chunk 200 400 \
#       --input-norm ${input_norm} \
#       --resnet-size ${resnet_size} \
#       --nj 12 \
#       --epochs 60 \
#       --batch-size ${batch_size} \
#       --optimizer ${optimizer} \
#       --scheduler ${scheduler} \
#       --lr 0.1 \
#       --base-lr 0.000006 \
#       --mask-layer ${mask_layer} \
#       --milestones 10,20,30,40,50 \
#       --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs${subset}_${mask_layer}/${loss}_${optimizer}_${scheduler}/${input_norm}_batch${batch_size}_${block_type}_down${downsample}_none1_${encoder_type}_dp1584_alpha${alpha}_em${embedding_size}_wd5e4_var \
#       --resume Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs${subset}_${mask_layer}/${loss}_${optimizer}_${scheduler}/${input_norm}_batch${batch_size}_${block_type}_down${downsample}_none1_${encoder_type}_dp1584_alpha${alpha}_em${embedding_size}_wd5e4_var/checkpoint_60.pth \
#       --kernel-size ${kernel} \
#       --downsample ${downsample} \
#       --channels 16,32,64,128 \
#       --fast none1 \
#       --stride 2,1 \
#       --block-type ${block_type} \
#       --embedding-size ${embedding_size} \
#       --time-dim 1 \
#       --avg-size 5 \
#       --encoder-type ${encoder_type} \
#       --num-valid 2 \
#       --alpha ${alpha} \
#       --margin 0.2 \
#       --s 30 \
#       --weight-decay 0.0005 \
#       --dropout-p 0.1584 \
#       --gpu-id 0,1 \
#       --extract \
#       --cos-sim \
#       --all-iteraion 0 \
#       --remove-vad \
#       --loss-type ${loss}

    python TrainAndTest/train_egs.py \
      --model ${model} \
      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev${subset}_fb${input_dim} \
      --train-test-dir ${lstm_dir}/data/${datasets}/${feat_type}/dev_fb${input_dim}/trials_dir \
      --train-trials trials_2w \
      --shuffle \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev${subset}_fb${input_dim}_valid \
      --test-dir ${lstm_dir}/data/${testset}/${feat_type}/test_fb${input_dim} \
      --feat-format kaldi \
      --random-chunk 200 400 \
      --input-norm ${input_norm} \
      --input-dim 40 \
      --resnet-size ${resnet_size} \
      --nj 12 \
      --epochs 30 \
      --batch-size ${batch_size} \
      --optimizer ${optimizer} \
      --scheduler ${scheduler} \
      --lr 0.001 \
      --base-lr 0.00001 \
      --mask-layer ${mask_layer} \
      --init-weight ${weight} \
      --scale ${scale} \
      --weight-p ${weight_p} \
      --milestones 10,20,30,40,50 \
      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs${subset}_${mask_layer}/${loss}_${optimizer}_${scheduler}/${input_norm}_batch${batch_size}_${block_type}_down${downsample}_${fast}_${encoder_type}_dp00_alpha${alpha}_em${embedding_size}_${weight}_scale${scale}dp${weight_p}_wd5e4_var \
      --resume Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs${subset}_${mask_layer}/${loss}_${optimizer}_${scheduler}/${input_norm}_batch${batch_size}_${block_type}_down${downsample}_${fast}_${encoder_type}_dp00_alpha${alpha}_em${embedding_size}_${weight}_scale${scale}dp${weight_p}_wd5e4_var/checkpoint_30.pth \
      --kernel-size ${kernel} \
      --downsample ${downsample} \
      --channels 16,32,64,128 \
      --fast ${fast} \
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
      --dropout-p 0 \
      --gpu-id 0,1 \
      --extract \
      --cos-sim \
      --all-iteraion 0 \
      --remove-vad \
      --loss-type ${loss}
  done
  exit
fi


if [ $stage -le 200 ]; then
  lstm_dir=/home/work2020/yangwenhao/project/lstm_speaker_verification
  datasets=aishell2
  testset=aishell2
  feat_type=klfb
  model=ThinResNet
  resnet_size=18
  encoder_type=SAP2
  embedding_size=256
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
  mask_layer=baseline
  weight=vox2_rcf
        # --milestones 15,25,35,45 \

  for encoder_type in SAP2; do
    echo -e "\n\033[1;4;31m Stage${stage}: Training ${model}${resnet_size} in ${datasets}_egs with ${loss} with ${input_norm} normalization \033[0m\n"
    python TrainAndTest/train_egs.py \
      --model ${model} \
      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev_fb${input_dim} \
      --train-test-dir ${lstm_dir}/data/${datasets}/${feat_type}/dev_fb${input_dim}/trials_dir \
      --train-trials trials_2w \
      --shuffle \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev_fb${input_dim}_valid \
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
      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_${mask_layer}/${loss}_${optimizer}_${scheduler}/${input_norm}_batch${batch_size}_${block_type}_down${downsample}_${fast}_${encoder_type}_dp01_alpha${alpha}_em${embedding_size}_wd5e4_var \
      --resume Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_${mask_layer}/${loss}_${optimizer}_${scheduler}/${input_norm}_batch${batch_size}_${block_type}_down${downsample}_${fast}_${encoder_type}_dp01_alpha${alpha}_em${embedding_size}_wd5e4_var/checkpoint_50.pth \
      --kernel-size ${kernel} \
      --downsample ${downsample} \
      --channels 16,32,64,128 \
      --fast ${fast} \
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
      --cos-sim \
      --all-iteraion 0 \
      --remove-vad \
      --loss-type ${loss}

    # python TrainAndTest/train_egs.py \
    #   --model ${model} \
    #   --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev_fb${input_dim} \
    #   --train-test-dir ${lstm_dir}/data/${datasets}/${feat_type}/dev_fb${input_dim}/trials_dir \
    #   --train-trials trials_2w \
    #   --shuffle \
    #   --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev_fb${input_dim}_valid \
    #   --test-dir ${lstm_dir}/data/${testset}/${feat_type}/test_fb${input_dim} \
    #   --feat-format kaldi \
    #   --random-chunk 200 400 \
    #   --input-norm ${input_norm} \
    #   --resnet-size ${resnet_size} \
    #   --nj 12 \
    #   --epochs 60 \
    #   --batch-size ${batch_size} \
    #   --optimizer ${optimizer} \
    #   --scheduler ${scheduler} \
    #   --lr 0.1 \
    #   --base-lr 0.000006 \
    #   --mask-layer ${mask_layer} \
    #   --init-weight ${weight} \
    #   --milestones 10,20,30 \
    #   --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_${mask_layer}/${loss}_${optimizer}_${scheduler}/chn32_${input_norm}_batch${batch_size}_${block_type}_down${downsample}_${fast}_${encoder_type}_dp01_alpha${alpha}_em${embedding_size}_${weight}_wd5e4_var \
    #   --resume Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_${mask_layer}/${loss}_${optimizer}_${scheduler}/chn32_${input_norm}_batch${batch_size}_${block_type}_down${downsample}_${fast}_${encoder_type}_dp01_alpha${alpha}_em${embedding_size}_${weight}_wd5e4_var/checkpoint_40.pth \
    #   --kernel-size ${kernel} \
    #   --downsample ${downsample} \
    #   --channels 32,64,128,256 \
    #   --fast ${fast} \
    #   --stride 2,1 \
    #   --block-type ${block_type} \
    #   --embedding-size ${embedding_size} \
    #   --time-dim 1 \
    #   --avg-size 5 \
    #   --encoder-type ${encoder_type} \
    #   --num-valid 2 \
    #   --alpha ${alpha} \
    #   --margin 0.2 \
    #   --s 30 \
    #   --weight-decay 0.0005 \
    #   --dropout-p 0.1 \
    #   --gpu-id 0,1 \
    #   --extract \
    #   --cos-sim \
    #   --all-iteraion 0 \
    #   --remove-vad \
    #   --loss-type ${loss}
  done
  exit
fi
