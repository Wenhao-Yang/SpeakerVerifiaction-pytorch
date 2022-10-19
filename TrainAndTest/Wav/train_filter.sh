#!/usr/bin/env bash

stage=20
waited=0
while [ $(ps 233979 | wc -l) -eq 2 ]; do
  sleep 60
  waited=$(expr $waited + 1)
  echo -en "\033[1;4;31m Having waited for ${waited} minutes!\033[0m\r"
done
lstm_dir=/home/yangwenhao/project/lstm_speaker_verification

if [ $stage -le 0 ]; then
  datasets=vox1
  model=TDNN_v5
  feat_type=spect
  feat=log
  block_type=None
  input_norm=Mean
  dropout_p=0
  encoder_type=STAP
  #  loss=arcsoft
  loss=soft
  avgsize=4
  alpha=0
  embedding_size=256
  block_type=None
  filter=fBPLayer
  feat_dim=24
  lr_ratio=1

  for filter in fLLayer; do
    echo -e "\n\033[1;4;31m Stage${stage} :Training ${model} in vox1 with ${loss} kernel 5,5 \033[0m\n"
    python TrainAndTest/train_egs.py \
      --model ${model} \
      --train-dir ${lstm_dir}/data/vox1/egs/spect/dev_${feat} \
      --train-test-dir ${lstm_dir}/data/vox1/spect/dev_${feat}/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/vox1/egs/spect/valid_${feat} \
      --test-dir ${lstm_dir}/data/vox1/spect/test_${feat} \
      --batch-size 128 \
      --input-norm ${input_norm} --test-input fix \
      --feat-format kaldi \
      --nj 10 --epochs 40 \
      --lr 0.1 \
      --input-dim 161 \
      --filter ${filter} \
      --time-dim 1 --exp \
      --feat-dim ${feat_dim} \
      --scheduler rop \
      --patience 3 \
      --milestones 10,20,30 \
      --check-path Data/checkpoint/${model}/${datasets}/${feat_type}_egs_filter/${loss}_0ce/Input${input_norm}_${encoder_type}_${block_type}_dp${dropout_p}_avg${avgsize}_alpha${alpha}_em${embedding_size}_wd5e4/${filter}${feat_dim}_adalr${lr_ratio}_full \
      --resume Data/checkpoint/${model}/${datasets}/${feat_type}_egs_filter/${loss}_0ce/Input${input_norm}_${encoder_type}_${block_type}_dp${dropout_p}_avg${avgsize}_alpha${alpha}_em${embedding_size}_wd5e4/${filter}${feat_dim}_adalr${lr_ratio}_full/checkpoint_9.pth \
      --stride 1 \
      --block-type ${block_type} \
      --channels 512,512,512,512,1500 \
      --encoder-type ${encoder_type} \
      --embedding-size ${embedding_size} \
      --avg-size ${avgsize} \
      --alpha ${alpha} \
      --num-valid 2 \
      --margin 0.25 --s 30 --m 3 \
      --lr-ratio ${lr_ratio} \
      --weight-decay 0.0005 \
      --dropout-p ${dropout_p} \
      --gpu-id 0,1 \
      --cos-sim \
      --extract \
      --all-iteraion 0 \
      --loss-type ${loss}
  done
  exit
fi


if [ $stage -le 10 ]; then
#  lstm_dir=/home/work2020/yangwenhao/project/lstm_speaker_verification
  datasets=aidata
  feat_type=wave
  model=ThinResNet
  resnet_size=18
  encoder_type=ASTP2
  embedding_size=256
  block_type=seblock
  downsample=k1
  kernel=5,5
  loss=arcsoft
  alpha=0
  input_norm=Mean
  mask_layer=baseline
  scheduler=rop
  optimizer=sgd
  input_dim=40
  batch_size=256
  power_weight=max

  expansion=4
  chn=16
  cyclic_epoch=8
  red_ratio=2
  avg_size=5
  fast=none1
  filter_layer=fbank
  feat_dim=40


  for resnet_size in 18; do
  for seed in 123456 123457 123458; do
    echo -e "\n\033[1;4;31m Stage${stage}: Training ${model}${resnet_size} in ${datasets}_egs with ${loss} with ${input_norm} normalization \033[0m\n"
    mask_layer=baseline
    weight=vox2_rcf
      #     --init-weight ${weight} \
      # --power-weight ${power_weight} \
      # _${weight}${power_weight}
    if [ $resnet_size -le 34 ];then
      expansion=1
      batch_size=256
    else
      expansion=2
      batch_size=256
      exp_str=_exp${expansion}
    fi
    if [ $chn -eq 16 ]; then
      channels=16,32,64,128
      chn_str=
    elif [ $chn -eq 32 ]; then
      channels=32,64,128,256
      chn_str=chn32_
    elif [ $chn -eq 64 ]; then
      channels=64,128,256,512
      chn_str=chn64_
    fi
    if [[ $mask_layer == attention* ]];then
      at_str=_${weight}
      if [[ $weight_norm != max ]];then
        at_str=${at_str}${weight_norm}
      fi
    elif [ "$mask_layer" = "drop" ];then
      at_str=_${weight}_dp${weight_p}s${scale}
    elif [ "$mask_layer" = "both" ];then
      at_str=_`echo $mask_len | sed  's/,//g'`
    else
      at_str=
    fi
    model_dir=${model}${resnet_size}/${datasets}/${feat_type}${input_dim}_egs_${mask_layer}/${loss}_${optimizer}_${scheduler}/${input_norm}_batch${batch_size}_${block_type}_red${red_ratio}${exp_str}_down${downsample}_avg${avg_size}_${encoder_type}_em${embedding_size}_dp01_alpha${alpha}_${fast}${at_str}_${chn_str}wd5e4_vares_bashuf2_${filter_layer}${feat_dim}/${seed}
    #
#
    python TrainAndTest/train_egs.py \
      --model ${model} --resnet-size ${resnet_size} \
      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/train \
      --train-test-dir ${lstm_dir}/data/${datasets}/test_10k \
      --train-trials trials \
      --shuffle --batch-shuffle --seed ${seed} \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/train_valid \
      --test-dir ${lstm_dir}/data/${datasets}/test_10k \
      --feat-format kaldi --nj 6 \
      --random-chunk 32000 64000 --chunk-size 48000 \
      --input-norm ${input_norm} --input-dim ${input_dim} \
      --feat-format wav \
      --filter ${filter_layer} --feat-dim ${feat_dim} \
      --epochs 60 --batch-size ${batch_size} \
      --optimizer ${optimizer} --scheduler ${scheduler} \
      --lr 0.1 --base-lr 0.000001 \
      --patience 3 --milestones 10,20,30,40 \
      --early-stopping --early-patience 20 --early-delta 0.01 --early-meta EER \
      --check-path Data/checkpoint/${model_dir} \
      --resume Data/checkpoint/${model_dir}/checkpoint_50.pth \
      --mask-layer ${mask_layer} \
      --kernel-size ${kernel} --channels ${channels} \
      --downsample ${downsample} --fast ${fast} --stride 2,1 \
      --block-type ${block_type} --red-ratio ${red_ratio} --expansion ${expansion} \
      --embedding-size ${embedding_size} \
      --time-dim 1 --avg-size ${avg_size} --encoder-type ${encoder_type} \
      --num-valid 2 \
      --alpha ${alpha} \
      --loss-type ${loss} --margin 0.2 --s 30 --all-iteraion 0 \
      --weight-decay 0.0005 \
      --dropout-p 0.1 \
      --gpu-id 0,6 \
      --extract --cos-sim \
      --remove-vad
  done
  done
  exit
fi

if [ $stage -le 11 ]; then
#  lstm_dir=/home/work2020/yangwenhao/project/lstm_speaker_verification
  datasets=aidata
  feat_type=wave
  model=ThinResNet
  resnet_size=18
  encoder_type=ASTP2
  embedding_size=256
  block_type=seblock
  downsample=k1
  kernel=5,5
  loss=arcsoft
  alpha=0
  input_norm=Mean
  mask_layer=baseline
  scheduler=rop
  optimizer=sgd
  input_dim=40
  batch_size=256
  power_weight=max

  expansion=4
  chn=16
  cyclic_epoch=8
  red_ratio=2
  avg_size=5
  fast=none1
  filter_layer=fbank
  feat_dim=40
  lamda_beta=0.2


  for lamda_beta in 0.5 1 2; do
  for seed in 123456 123457 123458; do
    echo -e "\n\033[1;4;31m Stage${stage}: Training ${model}${resnet_size} in ${datasets}_egs with ${loss} with ${input_norm} normalization \033[0m\n"
    mask_layer=baseline
    weight=vox2_rcf
      #     --init-weight ${weight} \
      # --power-weight ${power_weight} \
      # _${weight}${power_weight}
    if [ $resnet_size -le 34 ];then
      expansion=1
      batch_size=256
    else
      expansion=2
      batch_size=256
      exp_str=_exp${expansion}
    fi
    if [ $chn -eq 16 ]; then
      channels=16,32,64,128
      chn_str=
    elif [ $chn -eq 32 ]; then
      channels=32,64,128,256
      chn_str=chn32_
    elif [ $chn -eq 64 ]; then
      channels=64,128,256,512
      chn_str=chn64_
    fi
    if [[ $mask_layer == attention* ]];then
      at_str=_${weight}
      if [[ $weight_norm != max ]];then
        at_str=${at_str}${weight_norm}
      fi
    elif [ "$mask_layer" = "drop" ];then
      at_str=_${weight}_dp${weight_p}s${scale}
    elif [ "$mask_layer" = "both" ];then
      at_str=_`echo $mask_len | sed  's/,//g'`
    else
      at_str=
    fi
    model_dir=${model}${resnet_size}/${datasets}/${feat_type}${input_dim}_egs_${mask_layer}/${loss}_${optimizer}_${scheduler}/${input_norm}_batch${batch_size}_${block_type}_red${red_ratio}${exp_str}_down${downsample}_avg${avg_size}_${encoder_type}_em${embedding_size}_dp01_alpha${alpha}_${fast}${at_str}_${chn_str}wd5e4_vares_bashuf2_${filter_layer}${feat_dim}_mixup${lamda_beta}_2/${seed}
    #
#
    python TrainAndTest/train_egs_mixup.py \
      --model ${model} --resnet-size ${resnet_size} \
      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/train \
      --train-test-dir ${lstm_dir}/data/${datasets}/test_10k \
      --train-trials trials \
      --shuffle --batch-shuffle --seed ${seed} \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/train_valid \
      --test-dir ${lstm_dir}/data/${datasets}/test_10k \
      --feat-format kaldi --nj 6 \
      --random-chunk 32000 64000 --chunk-size 48000 \
      --input-norm ${input_norm} --input-dim ${input_dim} \
      --feat-format wav \
      --filter ${filter_layer} --feat-dim ${feat_dim} \
      --epochs 60 --batch-size ${batch_size} \
      --optimizer ${optimizer} --scheduler ${scheduler} \
      --lr 0.1 --base-lr 0.000001 \
      --patience 3 --milestones 10,20,30,40 \
      --early-stopping --early-patience 20 --early-delta 0.01 --early-meta EER \
      --check-path Data/checkpoint/${model_dir} \
      --resume Data/checkpoint/${model_dir}/checkpoint_50.pth \
      --mask-layer ${mask_layer} \
      --kernel-size ${kernel} --channels ${channels} \
      --downsample ${downsample} --fast ${fast} --stride 2,1 \
      --block-type ${block_type} --red-ratio ${red_ratio} --expansion ${expansion} \
      --embedding-size ${embedding_size} \
      --time-dim 1 --avg-size ${avg_size} --encoder-type ${encoder_type} \
      --num-valid 2 \
      --alpha ${alpha} \
      --loss-type ${loss} --margin 0.2 --s 30 --all-iteraion 0 \
      --weight-decay 0.0005 \
      --dropout-p 0.1 \
      --gpu-id 0,2 \
      --lamda-beta ${lamda_beta} \
      --extract --cos-sim \
      --remove-vad
  done
  done
  exit
fi

if [ $stage -le 12 ]; then
  datasets=aidata testset=aidata
  feat_type=wave
  model=ThinResNet resnet_size=18
  encoder_type=ASTP2
  embedding_size=256
  block_type=seblock
  red_ratio=2 expansion=1
  downsample=k1
  kernel=5,5
  loss=proser
  alpha=1
  input_norm=Mean
  mask_layer=baseline
  scheduler=rop
  optimizer=sgd
  input_dim=40
  filter_layer=fbank
  feat_dim=40
  batch_size=256
  fast=none1

  avg_size=5

#  encoder_type=SAP2
#  for input_dim in 64 80 ; do
  proser_ratio=1
  proser_gamma=0.01
  dummy=40

  for proser_gamma in 0.01 ; do
  for seed in 123458 ; do

    if [ $resnet_size -le 34 ];then
      expansion=1
      batch_size=256
    else
      expansion=2
      batch_size=256
      exp_str=_exp${expansion}
    fi
    model_dir=${model}${resnet_size}/${datasets}/${feat_type}${input_dim}_egs_${mask_layer}/${loss}_${optimizer}_${scheduler}/${input_norm}_batch${batch_size}_${block_type}_down${downsample}_${fast}_${encoder_type}_dp01_alpha${alpha}_em${embedding_size}_wd5e4_vares_bashuf2_${filter_layer}${feat_dim}_dummy${dummy}_beta${proser_ratio}_gamma${proser_gamma}/${seed}
    echo -e "\n\033[1;4;31m Stage${stage}: Training ${model}${resnet_size} in ${datasets}_egs with ${loss} with ${input_norm} normalization \033[0m\n"
    python TrainAndTest/train_egs_proser.py \
      --model ${model} --resnet-size ${resnet_size} \
      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/train \
      --train-test-dir ${lstm_dir}/data/${testset}/test_10k \
      --train-trials trials \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/train_valid \
      --test-dir ${lstm_dir}/data/${testset}/test_10k \
      --feat-format kaldi --shuffle --batch-shuffle \
      --random-chunk 32000 64000 --chunk-size 48000 \
      --feat-format wav \
      --filter ${filter_layer} --feat-dim ${feat_dim} \
      --input-dim ${input_dim} --input-norm ${input_norm} \
      --early-stopping --early-patience 20 --early-delta 0.01 --early-meta EER \
      --nj 6 --epochs 60 --batch-size ${batch_size} \
      --optimizer ${optimizer} --scheduler ${scheduler} --patience 3 \
      --lr 0.1 --base-lr 0.000001 \
      --mask-layer ${mask_layer} \
      --milestones 10,20,30,40,50 \
      --check-path Data/checkpoint/${model_dir} \
      --resume Data/checkpoint/${model_dir}/checkpoint_50.pth \
      --kernel-size ${kernel} --downsample ${downsample} \
      --channels 16,32,64,128 \
      --fast ${fast} --stride 2,1 \
      --block-type ${block_type} --red-ratio ${red_ratio} --expansion ${expansion} \
      --embedding-size ${embedding_size} \
      --time-dim 1 --avg-size ${avg_size} --dropout-p 0.1 \
      --encoder-type ${encoder_type} \
      --num-valid 2 \
      --alpha ${alpha} \
      --loss-type ${loss} --margin 0.2 --s 30 --all-iteraion 0 \
      --proser-ratio ${proser_ratio} --proser-gamma ${proser_gamma} --num-center ${dummy} \
      --weight-decay 0.0005 \
      --gpu-id 0 \
      --extract --cos-sim \
      --remove-vad
  done
  done
  exit
fi



if [ $stage -le 20 ]; then
  model=ResNet
  datasets=vox1
  #  feat=fb24
#  feat_type=pyfb
  feat_type=wave
  loss=arcsoft
  encod=ASTP2
  embedding_size=256
  # _lrr${lr_ratio}_lsr${loss_ratio}
  for lamda_beta in 2.0;do
    for seed in 123456 ; do
    for type in medium embedding ;do
     feat=fb${input_dim}

     echo -e "\n\033[1;4;31m Stage ${stage}: Training ${model}_${encod} in ${datasets}_${feat} with ${loss}\033[0m\n"
    #   CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 TrainAndTest/train_egs_dist.py
    #   CUDA_VISIBLE_DEVICES=3,5 python -m torch.distributed.launch --nproc_per_node=2 --master_port=417410 --nnodes=1 TrainAndTest/train_egs_dist.py --train-config=TrainAndTest/Fbank/ResNets/aidata_resnet.yaml --seed=${seed}
    #   CUDA_VISIBLE_DEVICES=5,6 python -m torch.distributed.launch --nproc_per_node=2 --master_port=417430 --nnodes=1 TrainAndTest/train_egs_dist.py --train-config=TrainAndTest/Wav/vox1_resnet.yaml --seed=${seed}
     CUDA_VISIBLE_DEVICES=5,6 python -m torch.distributed.launch --nproc_per_node=2 --master_port=417410 --nnodes=1 TrainAndTest/train_egs_dist_mixup.py --train-config=TrainAndTest/Wav/vox1_resnet_mixup_${type}.yaml --seed=${seed} --lamda-beta ${lamda_beta}
    done
    done
  done
  exit
fi