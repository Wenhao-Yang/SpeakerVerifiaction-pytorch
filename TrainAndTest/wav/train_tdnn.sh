#!/usr/bin/env bash

stage=10
waited=0
while [ $(ps 106034 | wc -l) -eq 2 ]; do
  sleep 60
  waited=$(expr $waited + 1)
  echo -en "\033[1;4;31m Having waited for ${waited} minutes!\033[0m\r"
done
lstm_dir=/home/yangwenhao/project/lstm_speaker_verification

if [ $stage -le 0 ]; then
  datasets=timit
  model=TDNN_v5
  feat_type=hst
  feat=c20
  block_type=basic
  input_norm=Mean
  dropout_p=0
  encoder_type=STAP
  #  loss=arcsoft
  loss=soft
  avgsize=4
  alpha=0
  embedding_size=128
  block_type=None
  feat_dim=40
  loss=soft
  scheduler=rop optimizer=sgd

  lr_ratio=0.1
#  --channels 512,512,512,512,1500 \

  for filter in sinc2down; do
    echo -e "\n\033[1;4;31m Stage${stage} :Training ${model} in ${datasets} with ${loss} kernel 5,5 \033[0m\n"
    model_dir=${model}/${datasets}/${feat_type}_egs_filter/${loss}_${optimizer}_${scheduler}/chn128_${input_norm}_${encoder_type}_${block_type}_dp${dropout_p}_alpha${alpha}_em${embedding_size}_wd5e4/${filter}${feat_dim}_bias2ddp054_adalr${lr_ratio}
    python TrainAndTest/train_egs.py \
      --model ${model} \
      --train-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/train_${feat}_down5 \
      --train-test-dir ${lstm_dir}/data/${datasets}/${feat_type}/train_${feat}/trials_dir \
      --train-trials trials_2w \
      --valid-dir ${lstm_dir}/data/${datasets}/egs/${feat_type}/train_${feat}_valid \
      --test-dir ${lstm_dir}/data/${datasets}/${feat_type}/test_${feat} \
      --batch-size 128 \
      --input-norm ${input_norm} --input-dim 40 \
      --test-input fix \
      --feat-format kaldi --nj 10 \
      --epochs 40 --patience 3 \
      --lr 0.1 \
      --random-chunk 6400 12800 \
      --chunk-size 9600 \
      --filter ${filter} --feat-dim ${feat_dim} \
      --optimizer ${optimizer} --scheduler ${scheduler} \
      --time-dim 1 --avg-size ${avgsize} \
      --milestones 10,20,30,40 \
      --check-path Data/checkpoint/${model_dir} \
      --resume Data/checkpoint/${model_dir}/checkpoint_9.pth \
      --stride 1 \
      --block-type ${block_type} \
      --channels 128,128,128,128,375 \
      --encoder-type ${encoder_type} \
      --embedding-size ${embedding_size} --alpha ${alpha} \
      --num-valid 2 \
      --margin 0.2 --s 30 --m 3 --all-iteraion 0 \
      --lr-ratio ${lr_ratio} \
      --filter-wd 0.001 --weight-decay 0.0005 \
      --dropout-p ${dropout_p} \
      --gpu-id 3 \
      --cos-sim --extract \
      --loss-type ${loss}
  done
  exit
fi


if [ $stage -le 10 ]; then
  model=ECAPA
  datasets=vox2
  #  feat=fb24
#  feat_type=pyfb
  feat_type=wave
  loss=arcsoft
  encod=ASTP2 embedding_size=256
  # _lrr${lr_ratio}_lsr${loss_ratio}
  for lamda_beta in 0.2;do
    for seed in 123456 123457 123458; do
    for type in 01 ;do
    #  feat=fb${input_dim}

     echo -e "\n\033[1;4;31m Stage ${stage}: Training ${model}_${encod} in ${datasets}_${feat} with ${loss}\033[0m\n"
    #   CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 TrainAndTest/train_egs_dist.py
    #   CUDA_VISIBLE_DEVICES=3,5 python -m torch.distributed.launch --nproc_per_node=2 --master_port=417410 --nnodes=1 TrainAndTest/train_egs_dist.py --train-config=TrainAndTest/Fbank/ResNets/aidata_resnet.yaml --seed=${seed}
      #  CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node=2 --master_port=41725 --nnodes=1 TrainAndTest/train_egs_dist.py --train-config=TrainAndTest/Wav/vox2_ecapa.yaml --seed=${seed}
      CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node=2 --master_port=41725 --nnodes=1 TrainAndTest/train_egs/train_dist.py --train-config=TrainAndTest/wav/vox2_int_original.yaml --seed=${seed}
      #  CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node=2 --master_port=41425 --nnodes=1 TrainAndTest/train_egs_dist_mixup.py --train-config=TrainAndTest/Wav/vox2_ecapa.yaml --seed=${seed} --lamda-beta ${lamda_beta}
#     CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 --master_port=417410 --nnodes=1 TrainAndTest/train_egs_dist_mixup.py --train-config=TrainAndTest/Wav/vox1_resnet_mixup_${type}.yaml --seed=${seed} --lamda-beta ${lamda_beta}
    done
    done
  done
  exit
fi