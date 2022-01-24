#!/usr/bin/env bash

stage=60

waited=0
while [ $(ps 17809 | wc -l) -eq 2 ]; do
  sleep 60
  waited=$(expr $waited + 1)
  echo -en "\033[1;4;31m Having waited for ${waited} minutes!\033[0m\r"
done
lstm_dir=/home/yangwenhao/project/lstm_speaker_verification

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
  block_type=basic
  embedding_size=256
  input_norm=Mean
  loss=soft
  feat_type=klsp
  sname=dev

  for sname in dev dev_aug_com; do
    echo -e "\n\033[1;4;31mStage ${stage}: Training ${model}${resnet_size} in ${datasets}_egs with ${loss} \033[0m\n"
    python TrainAndTest/Spectrogram/train_egs.py \
      --model ${model} \
      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname} \
      --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/dev/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname}_valid \
      --test-dir ${lstm_dir}/data/vox1/${feat_type}/test \
      --feat-format kaldi \
      --input-norm ${input_norm} \
      --resnet-size ${resnet_size} \
      --nj 12 \
      --epochs 60 \
      --scheduler rop \
      --patience 3 \
      --accu-steps 1 \
      --lr 0.1 \
      --milestones 10,20,40,50 \
      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_baseline/${loss}/${encoder_type}_em${embedding_size}_alpha${alpha}_wde3_${sname}_adam \
      --resume Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_baseline/${loss}/${encoder_type}_em${embedding_size}_alpha${alpha}_wde3_${sname}_adam/checkpoint_10.pth \
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
      --weight-decay 0.001 \
      --dropout-p 0 \
      --gpu-id 0,1 \
      --all-iteraion 0 \
      --extract \
      --cos-sim \
      --loss-type ${loss}
  done
  exit
fi

if [ $stage -le 21 ]; then
  datasets=vox1
  model=ThinResNet
  resnet_size=34
  encoder_type=AVG
  alpha=0
  block_type=basic_v2
  embedding_size=256
  input_norm=Mean
  loss=arcsoft
  feat_type=klsp
  sname=dev
  downsample=k3
  #        --scheduler cyclic \
#  for block_type in seblock cbam; do
    for downsample in k5; do
    echo -e "\n\033[1;4;31mStage ${stage}: Training ${model}${resnet_size} in ${datasets}_egs with ${loss} \033[0m\n"
    python TrainAndTest/train_egs.py \
      --model ${model} \
      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname} \
      --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/dev/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname}_valid \
      --test-dir ${lstm_dir}/data/vox1/${feat_type}/test \
      --feat-format kaldi \
      --input-norm ${input_norm} \
      --random-chunk 200 400 \
      --resnet-size ${resnet_size} \
      --downsample ${downsample} \
      --nj 12 \
      --epochs 50 \
      --optimizer sgd \
      --scheduler rop \
      --patience 2 \
      --accu-steps 1 \
      --fast none1 \
      --lr 0.1 \
      --milestones 10,20,30,40 \
      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_rvec/${loss}/input${input_norm}_${block_type}_down${downsample}_${encoder_type}_em${embedding_size}_dp125_alpha${alpha}_none1_vox2_wd5e4_var \
      --resume Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_rvec/${loss}/input${input_norm}_${block_type}_down${downsample}_${encoder_type}_em${embedding_size}_dp125_alpha${alpha}_none1_vox2_wd5e4_var/checkpoint_25.pth \
      --kernel-size 5,5 \
      --channels 16,32,64,128 \
      --input-dim 161 \
      --block-type ${block_type} \
      --red-ratio 8 \
      --stride 2,2 \
      --batch-size 128 \
      --embedding-size ${embedding_size} \
      --time-dim 1 \
      --avg-size 5 \
      --encoder-type ${encoder_type} \
      --num-valid 2 \
      --alpha ${alpha} \
      --margin 0.2 \
      --grad-clip 0 \
      --s 30 \
      --lr-ratio 0.01 \
      --weight-decay 0.0005 \
      --dropout-p 0.125 \
      --gpu-id 0,1 \
      --all-iteraion 0 \
      --extract \
      --shuffle \
      --cos-sim \
      --loss-type ${loss}
  done

  for downsample in k5; do
    echo -e "\n\033[1;4;31mStage ${stage}: Training ${model}${resnet_size} in ${datasets}_egs with ${loss} \033[0m\n"
    python TrainAndTest/train_egs.py \
      --model ${model} \
      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname} \
      --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/dev/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname}_valid \
      --test-dir ${lstm_dir}/data/vox1/${feat_type}/test \
      --feat-format kaldi \
      --input-norm ${input_norm} \
      --random-chunk 200 400 \
      --resnet-size ${resnet_size} \
      --downsample ${downsample} \
      --nj 12 \
      --epochs 50 \
      --optimizer sgd \
      --scheduler rop \
      --patience 2 \
      --mask-layer attention \
      --init-weight vox2 \
      --accu-steps 1 \
      --fast none1 \
      --lr 0.1 \
      --milestones 10,20,30,40 \
      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_rvec_attention/${loss}/input${input_norm}_${block_type}_down${downsample}_${encoder_type}_em${embedding_size}_dp125_alpha${alpha}_none1_vox2_wd5e4_var \
      --resume Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_rvec_attention/${loss}/input${input_norm}_${block_type}_down${downsample}_${encoder_type}_em${embedding_size}_dp125_alpha${alpha}_none1_vox2_wd5e4_var/checkpoint_25.pth \
      --kernel-size 5,5 \
      --channels 16,32,64,128 \
      --input-dim 161 \
      --block-type ${block_type} \
      --red-ratio 8 \
      --stride 2,2 \
      --batch-size 128 \
      --embedding-size ${embedding_size} \
      --time-dim 1 \
      --avg-size 5 \
      --encoder-type ${encoder_type} \
      --num-valid 2 \
      --alpha ${alpha} \
      --margin 0.2 \
      --grad-clip 0 \
      --s 30 \
      --lr-ratio 0.01 \
      --weight-decay 0.0005 \
      --dropout-p 0.125 \
      --gpu-id 0,1 \
      --shuffle \
      --all-iteraion 0 \
      --extract \
      --cos-sim \
      --loss-type ${loss}
  done

#  for sname in dev; do
#  for block_type in basic seblock cbam; do
#  for block_type in basic_v2 ; do
#    echo -e "\n\033[1;4;31mStage ${stage}: Training ${model}${resnet_size} in ${datasets}_egs with ${loss} \033[0m\n"
#    python TrainAndTest/train_egs.py \
#      --model ${model} \
#      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname} \
#      --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/dev/trials_dir \
#      --train-trials trials_2w \
#      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname}_valid \
#      --test-dir ${lstm_dir}/data/vox1/${feat_type}/test \
#      --feat-format kaldi \
#      --input-norm ${input_norm} \
#      --mask-layer attention \
#      --init-weight vox2 \
#      --random-chunk 200 400 \
#      --resnet-size ${resnet_size} \
#      --nj 12 \
#      --epochs 50 \
#      --optimizer sgd \
#      --scheduler rop \
#      --patience 2 \
#      --accu-steps 1 \
#      --lr 0.1 \
#      --milestones 10,20,30,40 \
#      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_rvec_attention/${loss}/input${input_norm}_${block_type}_${encoder_type}_em${embedding_size}_alpha${alpha}_dp25_vox2_wd5e4_var \
#      --resume Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_rvec_attention/${loss}/input${input_norm}_${block_type}_${encoder_type}_em${embedding_size}_alpha${alpha}_dp25_vox2_wd5e4_var/checkpoint_10.pth \
#      --kernel-size 5,5 \
#      --channels 16,32,64,128 \
#      --input-dim 161 \
#      --block-type ${block_type} \
#      --red-ratio 8 \
#      --stride 2 \
#      --batch-size 128 \
#      --embedding-size ${embedding_size} \
#      --time-dim 1 \
#      --avg-size 5 \
#      --encoder-type ${encoder_type} \
#      --num-valid 2 \
#      --alpha ${alpha} \
#      --margin 0.2 \
#      --grad-clip 0 \
#      --s 30 \
#      --lr-ratio 0.01 \
#      --weight-decay 0.0005 \
#      --dropout-p 0.25 \
#      --gpu-id 0,1 \
#      --all-iteraion 0 \
#      --extract \
#      --cos-sim \
#      --loss-type ${loss}
#  done
  exit
fi

if [ $stage -le 22 ]; then
  datasets=vox2
  model=LoResNet
  resnet_size=8
  encoder_type=None
  alpha=0
  block_type=cbam
  embedding_size=512
  input_norm=Mean
  loss=arcsoft
  dropout_p=0.1
  feat_type=klsp
  sname=dev

  for sname in dev; do
    echo -e "\n\033[1;4;31mStage ${stage}: Training ${model}${resnet_size} in ${datasets}_egs with ${loss} \033[0m\n"
    python TrainAndTest/train_egs.py \
      --model ${model} \
      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname} \
      --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/dev/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname}_valid \
      --test-dir ${lstm_dir}/data/vox1/${feat_type}/test \
      --feat-format kaldi \
      --input-norm ${input_norm} \
      --random-chunk 200 400 \
      --resnet-size ${resnet_size} \
      --nj 12 \
      --epochs 50 \
      --scheduler rop \
      --patience 2 \
      --accu-steps 1 \
      --lr 0.1 \
      --milestones 10,20,30,40 \
      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_baseline/${loss}/${encoder_type}_${block_type}_em${embedding_size}_alpha${alpha}_dp01_wde4_${sname}_var \
      --resume Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_baseline/${loss}/${encoder_type}_${block_type}_em${embedding_size}_alpha${alpha}_dp01_wde4_${sname}_var/checkpoint_5.pth \
      --channels 64,128,256 \
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
      --dropout-p ${dropout_p} \
      --gpu-id 0,1 \
      --all-iteraion 0 \
      --extract \
      --cos-sim \
      --loss-type ${loss}
  done
  exit
fi

if [ $stage -le 50 ]; then
  datasets=vox2
  model=ThinResNet
  resnet_size=34
  encoder_type=SAP2
  alpha=0
  block_type=cbam_v2
  embedding_size=256
  input_norm=Mean
  loss=arcsoft
  feat_type=klsp
  sname=dev
#  downsample=k3

  mask_layer=rvec
  scheduler=rop
  optimizer=sgd
  fast=none1

  #        --scheduler cyclic \
#  for block_type in seblock cbam; do
    for downsample in k5; do
    echo -e "\n\033[1;4;31mStage ${stage}: Training ${model}${resnet_size} in ${datasets}_egs with ${loss} \033[0m\n"
    python TrainAndTest/train_egs.py \
      --model ${model} \
      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname} \
      --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/dev/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname}_valid \
      --test-dir ${lstm_dir}/data/vox1/${feat_type}/test \
      --feat-format kaldi \
      --input-norm ${input_norm} \
      --random-chunk 200 400 \
      --optimizer ${optimizer} \
      --scheduler ${scheduler} \
      --resnet-size ${resnet_size} \
      --downsample ${downsample} \
      --nj 12 \
      --epochs 60 \
      --patience 3 \
      --accu-steps 1 \
      --fast ${fast} \
      --lr 0.1 \
      --milestones 10,20,30,40 \
      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_${mask_layer}/${loss}_${optimizer}_${scheduler}/input${input_norm}_${block_type}_down${downsample}_${encoder_type}_em${embedding_size}_dp01_alpha${alpha}_${fast}_wde4_var \
      --resume Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_${mask_layer}/${loss}_${optimizer}_${scheduler}/input${input_norm}_${block_type}_down${downsample}_${encoder_type}_em${embedding_size}_dp01_alpha${alpha}_${fast}_wde4_var/checkpoint_25.pth \
      --kernel-size 5,5 \
      --channels 16,32,64,128 \
      --input-dim 161 \
      --block-type ${block_type} \
      --red-ratio 8 \
      --stride 2,2 \
      --batch-size 128 \
      --embedding-size ${embedding_size} \
      --time-dim 1 \
      --avg-size 5 \
      --encoder-type ${encoder_type} \
      --num-valid 2 \
      --alpha ${alpha} \
      --margin 0.2 \
      --grad-clip 0 \
      --s 30 \
      --lr-ratio 0.01 \
      --weight-decay 0.0001 \
      --dropout-p 0.1 \
      --gpu-id 0,1 \
      --all-iteraion 0 \
      --extract \
      --shuffle \
      --cos-sim \
      --loss-type ${loss}
  done

fi

if [ $stage -le 60 ]; then
  datasets=vox2
  testset=vox1
  model=ThinResNet
  resnet_size=34
  encoder_type=SAP2
  alpha=0
  block_type=cbam_v2
  embedding_size=512
  input_norm=Mean
  loss=arcsoft
  feat_type=klsp
  sname=dev

  mask_layer=rvec
  scheduler=rop
  optimizer=sgd
  fast=none1
  downsample=k5

  for sname in dev ; do
    echo -e "\n\033[1;4;31mStage ${stage}: Training ${model}${resnet_size} in ${datasets}_egs with ${loss} \033[0m\n"
    python TrainAndTest/train_egs.py \
      --model ${model} \
      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname} \
      --train-test-dir ${lstm_dir}/data/${testset}/${feat_type}/${sname}/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/${sname}_valid \
      --test-dir ${lstm_dir}/data/${testset}/${feat_type}/test \
      --feat-format kaldi \
      --input-norm ${input_norm} \
      --resnet-size ${resnet_size} \
      --nj 12 \
      --epochs 60 \
      --random-chunk 200 400 \
      --optimizer ${optimizer} \
      --scheduler ${scheduler} \
      --patience 3 \
      --accu-steps 1 \
      --lr 0.1 \
      --milestones 10,20,40,50 \
      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_${mask_layer}/${loss}_${optimizer}_${scheduler}/chn32_${input_norm}_${block_type}_down${downsample}_${encoder_type}_em${embedding_size}_dp01_alpha${alpha}_${fast}_wde4_var \
      --resume Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_${mask_layer}/${loss}_${optimizer}_${scheduler}/chn32_${input_norm}_${block_type}_down${downsample}_${encoder_type}_em${embedding_size}_dp01_alpha${alpha}_${fast}_wde4_var/checkpoint_10.pth \
      --channels 32,64,128,256 \
      --downsample ${downsample} \
      --input-dim 161 \
      --fast ${fast} \
      --block-type ${block_type} \
      --stride 2 \
      --batch-size 128 \
      --embedding-size ${embedding_size} \
      --time-dim 1 \
      --avg-size 5 \
      --encoder-type ${encoder_type} \
      --num-valid 2 \
      --alpha ${alpha} \
      --margin 0.2 \
      --grad-clip 0 \
      --s 30 \
      --lr-ratio 0.01 \
      --weight-decay 0.0001 \
      --dropout-p 0.1 \
      --gpu-id 0,1,4 \
      --shuffle \
      --all-iteraion 0 \
      --extract \
      --cos-sim \
      --loss-type ${loss}
  done
  exit
fi