#!/usr/bin/env bash

stage=0
waited=0
while [ `ps 3492243 | wc -l` -eq 2 ]; do
  sleep 60
  waited=$(expr $waited + 1)
  echo -en "\033[1;4;31m Having waited for ${waited} minutes!\033[0m\r"
done
#stage=10

lstm_dir=/home/work2020/yangwenhao/project/lstm_speaker_verification

if [ $stage -le 0 ]; then
  model=ThinResNet
  datasets=aidata feat_type=klfb
  encod=SAP2 embedding_size=256
  input_dim=40 input_norm=Mean
  loss=arcsoft lr_ratio=0 loss_ratio=10
  subset=
  activation=leakyrelu
  scheduler=cyclic optimizer=adam
  stat_type=margin1 #margin1sum
  m=1.0
  seed=123457
  # _lrr${lr_ratio}_lsr${loss_ratio}
  # CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 TrainAndTest/train_egs_dist.py --train-config=TrainAndTest/Fbank/ResNets/vox1_aug/vox1_clsaug5.yaml --seed=${seed}

  # seed=123456
  for lamda_beta in 0.2 ; do
  for seed in 123456 123457 123458 ; do
   for layer in 7 ; do
   echo -e "\n\033[1;4;31m Stage ${stage}: Training ${model}_${encod} in ${datasets}_${feat} with ${loss}\033[0m\n"
    # CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 TrainAndTest/train_egs/train_egs_dist.py --train-config=TrainAndTest/wav/resnet/aidata_float.yaml --seed=${seed}
    # sleep 5
    # CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 TrainAndTest/train_egs/train_egs_dist.py --train-config=TrainAndTest/wav/resnet/aidata_int.yaml --seed=${seed}

    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 TrainAndTest/train_egs/train_dist.py --train-config=TrainAndTest/wav/resnet/aidata_float_original.yaml --seed=${seed}


  done
  done
  done
  exit
fi
