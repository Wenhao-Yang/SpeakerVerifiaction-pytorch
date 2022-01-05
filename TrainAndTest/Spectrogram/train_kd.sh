#!/usr/bin/env bash

stage=0
waited=0
while [ $(ps 27253 | wc -l) -eq 2 ]; do
  sleep 60
  waited=$(expr $waited + 1)
  echo -en "\033[1;4;31m Having waited for ${waited} minutes!\033[0m\r"
done
#stage=10
lstm_dir=/home/work2020/yangwenhao/project/lstm_speaker_verification

#if [ $stage -le 0 ]; then
#
#
#fi

if [ $stage -le 0 ]; then
  datasets=vox1
  feat_type=klsp
  model=LoResNet
  resnet_size=8
  encoder_type=AVG
  embedding_size=256
  block_type=cbam
  kernel=5,5
  loss=arcsoft
  alpha=0
  input_norm=Mean
  mask_layer=kd
  scheduler=rop
  optimizer=sgd
  nj=8

  teacher_dir=Data/checkpoint/LoResNet8/vox1/klsp_egs_baseline/arcsoft/None_cbam_em256_alpha0_dp25_wd5e4_dev_var

   for encoder_type in AVG ; do
     echo -e "\n\033[1;4;31m Stage${stage}: Training ${model}${resnet_size} in ${datasets}_egs with ${loss} with ${input_norm} normalization \033[0m\n"
     python TrainAndTest/train_egs_kd.py \
       --model ${model} \
       --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev \
       --train-test-dir ${lstm_dir}/data/${datasets}/${feat_type}/dev/trials_dir \
       --train-trials trials_2w \
       --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev_valid \
       --test-dir ${lstm_dir}/data/${datasets}/${feat_type}/test \
       --feat-format kaldi \
       --random-chunk 200 400 \
       --input-norm ${input_norm} \
       --resnet-size ${resnet_size} \
       --nj ${nj} \
       --shuffle \
       --epochs 50 \
       --batch-size 128 \
       --optimizer ${optimizer} \
       --scheduler ${scheduler} \
       --lr 0.1 \
       --base-lr 0.000006 \
       --mask-layer ${mask_layer} \
       --milestones 10,20,30,40 \
       --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_${mask_layer}/${loss}_${optimizer}_${scheduler}/${input_norm}_${block_type}_${encoder_type}_dp20_alpha${alpha}_em${embedding_size}_wd5e4_chn16_var \
       --resume Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_${mask_layer}/${loss}_${optimizer}_${scheduler}/${input_norm}_${block_type}_${encoder_type}_dp20_alpha${alpha}_em${embedding_size}_wd5e4_chn16_var/checkpoint_50.pth \
       --kernel-size ${kernel} \
       --channels 16,32,64 \
       --stride 2 \
       --block-type ${block_type} \
       --embedding-size ${embedding_size} \
       --time-dim 1 \
       --avg-size 4 \
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
       --loss-type ${loss} \
       --distil-weight 0.5 \
       --teacher-model-yaml ${teacher_dir}/model.2022.01.05.yaml \
       --teacher-resume ${teacher_dir}/checkpoint_40.pth \
       --temperature 20
   done

#  weight=vox2
#  scale=0.2
#
#  for mask_layer in drop; do
##    mask_layer=baseline
#    echo -e "\n\033[1;4;31m Stage${stage}: Training ${model}${resnet_size} in ${datasets}_egs_${mask_layer} with ${loss} with ${input_norm} normalization \033[0m\n"
#    python TrainAndTest/train_egs.py \
#      --model ${model} \
#      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev \
#      --train-test-dir ${lstm_dir}/data/vox1/${feat_type}/dev/trials_dir \
#      --train-trials trials_2w \
#      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/dev_valid \
#      --test-dir ${lstm_dir}/data/vox1/${feat_type}/test \
#      --feat-format kaldi \
#      --random-chunk 200 400 \
#      --input-norm ${input_norm} \
#      --resnet-size ${resnet_size} \
#      --nj ${nj} \
#      --epochs 50 \
#      --batch-size 128 \
#      --shuffle \
#      --optimizer ${optimizer} \
#      --scheduler ${scheduler} \
#      --lr 0.1 \
#      --base-lr 0.000005 \
#      --mask-layer ${mask_layer} \
#      --init-weight ${weight} \
#      --scale ${scale} \
#      --milestones 10,20,30,40 \
#      --check-path Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_${mask_layer}/${loss}_${optimizer}_${scheduler}/${input_norm}_${block_type}_${encoder_type}_dp01_alpha${alpha}_em${embedding_size}_${weight}_wd5e4_var \
#      --resume Data/checkpoint/${model}${resnet_size}/${datasets}/${feat_type}_egs_${mask_layer}/${loss}_${optimizer}_${scheduler}/${input_norm}_${block_type}_${encoder_type}_dp01_alpha${alpha}_em${embedding_size}_${weight}_wd5e4_var/checkpoint_50.pth \
#      --kernel-size ${kernel} \
#      --channels 64,128,256 \
#      --stride 2 \
#      --block-type ${block_type} \
#      --embedding-size ${embedding_size} \
#      --time-dim 1 \
#      --avg-size 4 \
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
#      --loss-type ${loss}
#  done

  exit
fi
